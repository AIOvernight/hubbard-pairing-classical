"""Lattice geometry, gate layout and single-particle propagators used by the hierarchy.

Conventions
-----------
* Sites are numbered ``p = x + Lx*y``; fermionic modes are site-major,
  ``mode = 2*p + spin`` with spin 0 = up, 1 = down.
* Boundary signs ``(bx, by)`` multiply every hopping gate that crosses the
  x (resp. y) seam: PP=(1,1), AP=(-1,1), PA=(1,-1), AA=(-1,-1).
* The two-step circuit is ``prep -> G1 -> D(3) -> G2 -> D(1.5) -> measure``
  with Gaussian hopping layers G1, G2 (dt = 0.375) and onsite layers D(theta).
  The coefficients below are the implementation (state-propagator) convention
  of the circuit translation; they must not be replaced by physical
  Hamiltonian signs.

This module is self-contained so that the hierarchy can be used independently
of the circuit-translation and exact-simulation modules of the package.
"""
from functools import lru_cache

import numpy as np

# Preparation parameters (rounded values of the supplied circuit).
SPIN_LAYERS = (('Z', .12060681), ('X', .08211975), ('Y', .05188793), ('X', .33385306))
WEAK_HOP = .10667747          # weak preparation hopping angle
PREP_ONSITE = -2 * .48897462  # two onsite layers inside the preparation step
SPIN_HOP = -.04714107         # staggered local up<->down rotation
PREP_FINAL_ONSITE = 1.5       # final preparation onsite layer

# Two-step pulse suffix.
PULSE_DT = .375
G1 = (-.7511, -.47413)        # (k, j) generator of the first Gaussian layer
G2 = (.7511, -.47413)         # (k, j) generator of the second Gaussian layer
CENTRAL_ONSITE = 3.0          # D(3) between G1 and G2
TERMINAL_ONSITE = 1.5         # D(1.5) after G2 (commutes with A, cancels)

BOUNDARIES = {'PP': (1, 1), 'AP': (-1, 1), 'PA': (1, -1), 'AA': (-1, -1)}


def site(x, y, lx, ly):
    return x % lx + lx * (y % ly)


def stagger(p, lx):
    """(-1)^(x+y) for site p."""
    return (-1) ** (p % lx + p // lx)


def gate_layout(lx, ly):
    """Ordered hopping gates of one Trotter step.

    Yields ``(round, mode_a, mode_b, wraps_y, wraps_x, current_sign)``.
    Within a round the plaquettes are disjoint; the four edges of a plaquette
    are applied in the listed order.
    """
    for rnd in (0, 1):
        for species in (0, 1):
            for y0 in range(0, ly, 2):
                for x0 in range(0, lx, 2):
                    x, y = (x0, y0) if rnd == species else (x0 + 1, y0 + 1)
                    if rnd == species:
                        q = [site(x, y + 1, lx, ly), site(x, y, lx, ly), site(x + 1, y, lx, ly), site(x + 1, y + 1, lx, ly)]
                        ps = 1
                    else:
                        q = [site(x + 1, y, lx, ly), site(x + 1, y + 1, lx, ly), site(x, y + 1, lx, ly), site(x, y, lx, ly)]
                        ps = -1
                    for i, j, vertical, direction in [(0, 1, True, -1), (2, 3, True, 1), (1, 2, False, 1), (0, 3, False, 1)]:
                        yield (rnd, 2 * q[i] + species, 2 * q[j] + species,
                               vertical and y == ly - 1, (not vertical) and x == lx - 1, direction * ps)


def spin_bonds(lx, ly):
    """Nearest-neighbour bonds of the ZZ/YY spin layers of the preparation."""
    s = lambda x, y: site(x, y, lx, ly)
    return ([(s(2 * j, y), s(2 * j + 1, y)) for j in range(lx // 2) for y in range(ly)]
            + [(s(x, 2 * j), s(x, 2 * j + 1)) for x in range(lx) for j in range(ly // 2)]
            + [(s(2 * j + 1, y), s(2 * j + 2, y)) for j in range(lx // 2) for y in range(ly)]
            + [(s(x, 2 * j + 1), s(x, 2 * j + 2)) for x in range(lx) for j in range(ly // 2)])


@lru_cache(None)
def hopping_matrix(lx, ly, bx, by, k, j, dt=PULSE_DT):
    """Single-particle propagator W of one Gaussian layer (all hops, both rounds).

    Row a of W gives the coefficients of ``G^dagger c_a G`` in terms of the
    annihilators before the layer.
    """
    w = np.eye(2 * lx * ly, dtype=complex)
    for _, a, b, wy, wx, cs in gate_layout(lx, ly):
        z = dt * (bx if wx else 1) * (by if wy else 1) * (k + 1j * j * cs)
        r = abs(z)
        if not r:
            continue
        f = np.sin(r) * z / r
        x, y = w[a].copy(), w[b].copy()
        w[a] = np.cos(r) * x - 1j * f * y
        w[b] = np.cos(r) * y - 1j * np.conj(f) * x
    w.flags.writeable = False
    return w


def first_propagator(lx, ly, bx, by):
    """W1, the exact first Gaussian layer used to project preparation cumulants."""
    return hopping_matrix(lx, ly, bx, by, *G1)


def second_propagator(lx, ly, bx, by):
    """W2, the second Gaussian layer through which the observable is pulled back."""
    return hopping_matrix(lx, ly, bx, by, *G2)
