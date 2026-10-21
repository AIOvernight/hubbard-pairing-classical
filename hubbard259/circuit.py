"""Gate schedule and parameters of the two-step pairing circuit.

The physical (fermionic) content of the authors' 90-qubit two-step circuit is
the following sequence of layers acting on a half-filled ``lx x ly`` Hubbard
lattice (``U/t = 8``, Trotter step ``dt = 0.375``).  Mode ordering, edge order
and boundary signs are defined in :mod:`hubbard259.lattice`.

Elementary gates (all exactly number conserving)
------------------------------------------------
``hop(a, b, z)``
    ``exp[-i (z c_a^dag c_b + conj(z) c_b^dag c_a)]``.  On a single particle
    ``psi_a -> cos|z| psi_a - i f psi_b`` and
    ``psi_b -> cos|z| psi_b - i conj(f) psi_a`` with ``f = sin|z| z/|z|``.
``onsite(theta)``  (written ``D(theta)``)
    ``exp[-i theta sum_s (n_s,up - 1/2)(n_s,down - 1/2)]``.  ``D(1.5)`` is a
    half interaction step ``dt*U/2``, ``D(3)`` a full step.
Edge amplitude of a hopping layer with parameters ``(dt, k, j)``
    ``z = dt * edge_sign(edge, bx, by) * (k + 1j * j * edge.orientation)``.

Schedule (:func:`schedule`)
---------------------------
1. Spin preparation on one spin-1/2 per site (bit 0 = up, 1 = down):
   X-Neel product state (site spinor ``(|up> - s|down>)/sqrt2`` with
   ``s = stagger``), then ``exp(-i 0.12060681 ZZ)`` on all
   :func:`~hubbard259.lattice.spin_bonds`, staggered
   ``exp(-i 0.08211975 s X)``, ``exp(-i 0.05188793 YY)`` on all bonds,
   staggered ``exp(-i 0.33385306 s X)``.
2. Injection: spin configuration -> singly occupied Fock state
   (bit ``b`` of site ``s`` occupies mode ``2s+b``).
3. Weak preparation layers: hopping round 0 with real amplitude
   ``0.10667747`` (times boundary sign), ``D(-0.97794924)``, hopping round 1,
   ``D(-0.97794924)``, staggered spin mixing ``hop(2s, 2s+1, -0.04714107 s)``.
4. ``D(1.5)``.
5. Pulse 1: Gaussian hopping rounds 0 and 1 with ``(k, j) = (-0.7511, -0.47413)``,
   then ``D(3)``.
6. Pulse 2: Gaussian hopping rounds 0 and 1 with ``(k, j) = (+0.7511, -0.47413)``,
   then ``D(1.5)``.

Measured quantity (positive convention used throughout this package)
-------------------------------------------------------------------
``A = (2/N) [<D> - <eta^dag eta>]`` with ``D = sum_s n_s,up n_s,down`` and
``eta = sum_s stagger(s) c_s,down c_s,up``; ``N = lx*ly``.  This is the
quantity estimated by the authors' raw-shot estimator (two-step hardware
value ``A = 0.2658 +/- 0.0255``).  Taken literally, the paper's displayed convention gives ``P = -A`` and
the expression written in the tracker issue equals ``-A/2``.

Public API
----------
``DT, U, PREPARATION, PULSES, Layer, schedule, edge_amplitude,
hopping_matrix, pairing_from_moments, paper_convention, issue_convention``
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import Edge, edge_sign, plaquette_edges, stagger

__all__ = [
    "DT", "PREPARATION", "PULSES", "U", "Layer", "edge_amplitude",
    "hopping_matrix", "issue_convention", "pairing_from_moments",
    "paper_convention", "schedule",
]

#: Trotter step (units of 1/t) and interaction strength.
DT = 0.375
U = 8.0

#: Rounded preparation parameters exactly as they appear in the circuit.
PREPARATION = {
    "spin_zz": 0.12060681,
    "spin_x1": 0.08211975,
    "spin_yy": 0.05188793,
    "spin_x2": 0.33385306,
    "hopping": 0.10667747,
    "onsite": -2 * 0.48897462,
    "spin_mixing": -0.04714107,
    "onsite_half_step": DT * U / 2,
}

#: Gaussian pulse layers: ``(k, j, onsite angle applied after the pulse)``.
PULSES = ((-0.7511, -0.47413, DT * U), (0.7511, -0.47413, DT * U / 2))


@dataclass(frozen=True)
class Layer:
    """One layer of the schedule.

    ``kind`` is one of ``'neel'``, ``'spin_zz'``, ``'spin_x'``, ``'spin_yy'``,
    ``'inject'``, ``'hopping'``, ``'onsite'``, ``'spin_mixing'``.
    ``angle`` is the rotation angle of spin/onsite/spin-mixing layers.
    Hopping layers use ``round``, ``dt``, ``k``, ``j`` (see
    :func:`edge_amplitude`).  ``stage`` is ``'spin'``, ``'weak'``,
    ``'pulse1'`` or ``'pulse2'``.
    """
    kind: str
    stage: str
    angle: float = 0.0
    round: int = -1
    dt: float = 1.0
    k: float = 0.0
    j: float = 0.0


def schedule() -> tuple[Layer, ...]:
    """Ordered layers of the two-step circuit (see module docstring)."""
    p = PREPARATION
    layers = [
        Layer("neel", "spin"),
        Layer("spin_zz", "spin", p["spin_zz"]),
        Layer("spin_x", "spin", p["spin_x1"]),
        Layer("spin_yy", "spin", p["spin_yy"]),
        Layer("spin_x", "spin", p["spin_x2"]),
        Layer("inject", "spin"),
        Layer("hopping", "weak", round=0, dt=1.0, k=p["hopping"]),
        Layer("onsite", "weak", p["onsite"]),
        Layer("hopping", "weak", round=1, dt=1.0, k=p["hopping"]),
        Layer("onsite", "weak", p["onsite"]),
        Layer("spin_mixing", "weak", p["spin_mixing"]),
        Layer("onsite", "weak", p["onsite_half_step"]),
    ]
    for n, (k, j, theta) in enumerate(PULSES, start=1):
        for rnd in (0, 1):
            layers.append(Layer("hopping", f"pulse{n}", round=rnd, dt=DT, k=k, j=j))
        layers.append(Layer("onsite", f"pulse{n}", theta))
    return tuple(layers)


def edge_amplitude(edge: Edge, bx: int, by: int, layer: Layer) -> complex:
    """Complex hopping amplitude ``z`` of ``edge`` in a hopping ``layer``."""
    return layer.dt * edge_sign(edge, bx, by) * (layer.k + 1j * layer.j * edge.orientation)


def spin_mixing_amplitude(site: int, lx: int, layer: Layer) -> complex:
    """Amplitude of ``hop(2s, 2s+1, z)`` in the staggered spin-mixing layer."""
    return complex(layer.angle * stagger(site, lx))


def hopping_matrix(lx: int, ly: int, bx: int, by: int, k: float, j: float,
                   dt: float = DT) -> np.ndarray:
    """Single-particle matrix ``W`` of one Gaussian pulse (both rounds).

    ``W`` (``2N x 2N``, site-major modes) maps single-particle amplitudes,
    ``psi -> W psi``; equivalently ``G^dag c_a G = sum_b W_ab c_b``
    and ``G^dag c_a^dag G = sum_b conj(W_ab) c_b^dag``.
    """
    layer = Layer("hopping", "pulse", dt=dt, k=k, j=j)
    w = np.eye(2 * lx * ly, dtype=complex)
    for edge in plaquette_edges(lx, ly):   # already ordered round 0, round 1
        z = edge_amplitude(edge, bx, by, layer)
        r = abs(z)
        if not r:
            continue
        f = np.sin(r) * z / r
        x, y = w[edge.a].copy(), w[edge.b].copy()
        w[edge.a] = np.cos(r) * x - 1j * f * y
        w[edge.b] = np.cos(r) * y - 1j * np.conj(f) * x
    return w


def pairing_from_moments(doublons: float, eta_norm2: float, nsites: int) -> float:
    """Positive-convention observable ``A = 2 (<D> - <eta^dag eta>) / N``."""
    return 2.0 * (doublons - eta_norm2) / nsites


def paper_convention(a: float) -> float:
    """Pairing value in the paper's sign convention, ``P = -A``."""
    return -a


def issue_convention(a: float) -> float:
    """Value of the expression written in the tracker issue, ``-A/2``."""
    return -a / 2
