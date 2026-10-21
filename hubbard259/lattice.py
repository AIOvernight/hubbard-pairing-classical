"""Lattice geometry, mode ordering and boundary conditions.

Conventions (used by every module of the package)
-------------------------------------------------
* Sites of an ``lx x ly`` torus are numbered ``s = x + lx*y`` with
  ``0 <= x < lx`` and ``0 <= y < ly``; both ``lx`` and ``ly`` must be even.
* Fermionic modes are **site-major**: ``mode(s, spin) = 2*s + spin`` with
  ``spin = 0`` (up) and ``spin = 1`` (down).  Jordan-Wigner strings follow this
  mode order.
* The staggering sign is ``stagger(s, lx) = (-1)**(x + y)``.
* Boundary sectors are labelled by the signs ``(bx, by)`` attached to hopping
  terms that cross the x seam (``x = lx-1 -> 0``) and the y seam
  (``y = ly-1 -> 0``).  Names give the x boundary first:
  ``PP=(+1,+1)``, ``AP=(-1,+1)``, ``PA=(+1,-1)``, ``AA=(-1,-1)``.
  The "mixed" boundary condition of the target circuit is the uniform average
  over these four sectors.

Hopping geometry
----------------
The hopping part of one Trotter step is applied as two *rounds* of
plaquette gates.  In round ``r`` spin species ``sigma`` uses the even
plaquettes (lower-left corner ``(2j, 2k)``) when ``r == sigma`` and the odd
plaquettes (lower-left corner ``(2j+1, 2k+1)``) otherwise.  Each plaquette
contributes four nearest-neighbour edges.  :func:`plaquette_edges` returns
them as :class:`Edge` records in the order in which the gates are applied;
``Edge.orientation`` (+1/-1) is the direction of the bond current used by the
complex (Peierls-phase) hopping amplitudes of the pulse, and
``Edge.wraps_x`` / ``Edge.wraps_y`` mark seam-crossing edges that receive the
boundary signs ``bx`` / ``by``.

Public API
----------
``site_index, coordinates, stagger, mode, BOUNDARIES, BOUNDARY_NAMES,
boundary_name, Edge, plaquette_edges, edge_sign, spin_bonds, check_shape``
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BOUNDARIES", "BOUNDARY_NAMES", "Edge", "boundary_name", "check_shape",
    "coordinates", "edge_sign", "mode", "plaquette_edges", "site_index",
    "spin_bonds", "stagger",
]

#: Boundary sectors ``name -> (bx, by)``; x boundary is the first letter.
BOUNDARIES = {"PP": (1, 1), "AP": (-1, 1), "PA": (1, -1), "AA": (-1, -1)}
#: Canonical order of the four sectors.
BOUNDARY_NAMES = ("PP", "AP", "PA", "AA")


def boundary_name(bx: int, by: int) -> str:
    """Return ``'PP'``, ``'AP'``, ``'PA'`` or ``'AA'`` for signs ``(bx, by)``."""
    return ("P" if bx == 1 else "A") + ("P" if by == 1 else "A")


def check_shape(lx: int, ly: int) -> None:
    """Raise ``ValueError`` unless both dimensions are even and positive."""
    if lx < 2 or ly < 2 or lx % 2 or ly % 2:
        raise ValueError(f"lattice {lx}x{ly}: both dimensions must be even and >= 2")


def site_index(x: int, y: int, lx: int, ly: int) -> int:
    """Site number of (periodically wrapped) coordinates ``(x, y)``."""
    return x % lx + lx * (y % ly)


def coordinates(site: int, lx: int) -> tuple[int, int]:
    """Inverse of :func:`site_index`."""
    return site % lx, site // lx


def stagger(site: int, lx: int) -> int:
    """Checkerboard sign ``(-1)**(x+y)`` of a site."""
    x, y = coordinates(site, lx)
    return -1 if (x + y) % 2 else 1


def mode(site: int, spin: int) -> int:
    """Site-major fermionic mode index (spin 0 = up, 1 = down)."""
    return 2 * site + spin


@dataclass(frozen=True)
class Edge:
    """One hopping gate between fermionic modes ``a`` and ``b`` (same spin).

    ``round``: 0 or 1, the hopping round inside a Trotter step.
    ``wraps_y`` / ``wraps_x``: the edge crosses the y / x seam.
    ``orientation``: +1 or -1, sign of the current term ``j`` for this edge.
    """
    round: int
    a: int
    b: int
    wraps_y: bool
    wraps_x: bool
    orientation: int

    @property
    def spin(self) -> int:
        return self.a % 2

    @property
    def sites(self) -> tuple[int, int]:
        return self.a // 2, self.b // 2


def plaquette_edges(lx: int, ly: int) -> list[Edge]:
    """All hopping edges of one Trotter step, in gate order.

    Order: round 0 then round 1; inside a round spin up then spin down;
    plaquettes row by row; per plaquette the edge order
    (left vertical, right vertical, bottom/top horizontal ...) fixed below.
    """
    check_shape(lx, ly)
    out = []
    site = lambda x, y: site_index(x, y, lx, ly)  # noqa: E731
    for rnd in (0, 1):
        for spin in (0, 1):
            for k in range(ly // 2):
                for j in range(lx // 2):
                    if rnd == spin:
                        x, y = 2 * j, 2 * k
                        q = [site(x, y + 1), site(x, y), site(x + 1, y), site(x + 1, y + 1)]
                        ps = 1
                    else:
                        x, y = 2 * j + 1, 2 * k + 1
                        q = [site(x + 1, y), site(x + 1, y + 1), site(x, y + 1), site(x, y)]
                        ps = -1
                    for i, h, vertical, direction in ((0, 1, True, -1), (2, 3, True, 1),
                                                      (1, 2, False, 1), (0, 3, False, 1)):
                        out.append(Edge(rnd, mode(q[i], spin), mode(q[h], spin),
                                        vertical and y == ly - 1,
                                        (not vertical) and x == lx - 1,
                                        direction * ps))
    return out


def edge_sign(edge: Edge, bx: int, by: int) -> int:
    """Boundary sign of an edge in sector ``(bx, by)``."""
    return (bx if edge.wraps_x else 1) * (by if edge.wraps_y else 1)


def spin_bonds(lx: int, ly: int) -> list[tuple[int, int]]:
    """Nearest-neighbour site pairs of the initial spin-model layers.

    Four brick layers in this order: even x-bonds, even y-bonds, odd x-bonds,
    odd y-bonds (periodic).  Used for the ZZ and YY layers of the
    preparation.
    """
    check_shape(lx, ly)
    site = lambda x, y: site_index(x, y, lx, ly)  # noqa: E731
    return ([(site(2 * j, y), site(2 * j + 1, y)) for j in range(lx // 2) for y in range(ly)]
            + [(site(x, 2 * j), site(x, 2 * j + 1)) for x in range(lx) for j in range(ly // 2)]
            + [(site(2 * j + 1, y), site(2 * j + 2, y)) for j in range(lx // 2) for y in range(ly)]
            + [(site(x, 2 * j + 1), site(x, 2 * j + 2)) for x in range(lx) for j in range(ly // 2)])
