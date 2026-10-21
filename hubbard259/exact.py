"""Exact fixed-particle-number state-vector simulation of the circuit (CPU).

The state is stored on the half-filled sector (``N`` fermions in ``2N``
site-major modes, all magnetizations) as a complex128 vector indexed by the
sorted list of occupation bit masks.  Every gate of
:func:`hubbard259.circuit.schedule` is applied exactly; nothing is truncated
or expanded.  The CPU backend is intended for small lattices (4x2: 12870
amplitudes, well under a second; 6x2: 2.7 million amplitudes).
:mod:`hubbard259.exact_gpu` provides the same backend interface on a GPU for
16-site lattices (4x4, 8x2).

Usage::

    from hubbard259.exact import simulate
    r = simulate(4, 2, bx=1, by=1)       # dict with 'A', 'doublons', ...

Backend interface used by :func:`evolve`: ``load(occupations, amplitudes)``,
``hop(a, b, z)``, ``onsite(theta)``, ``measure() -> dict``.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import circuit
from .lattice import BOUNDARIES, check_shape, plaquette_edges, spin_bonds, stagger

__all__ = ["CPUBackend", "evolve", "initial_spin_state", "simulate", "simulate_sectors"]


def _parity(x: np.ndarray) -> np.ndarray:
    """Parity of the number of set bits (vectorized, uint64 input)."""
    if hasattr(np, "bitwise_count"):
        return (np.bitwise_count(x) & 1).astype(np.int8)
    x = x.copy()
    for shift in (32, 16, 8, 4, 2, 1):
        x ^= x >> np.uint64(shift)
    return (x & np.uint64(1)).astype(np.int8)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.array([int(v).bit_count() for v in x], dtype=np.uint8)


def initial_spin_state(lx: int, ly: int, layers=None):
    """Spin-preparation part of the schedule, followed by injection.

    Returns ``(occupations, amplitudes)``: fermionic occupation masks
    (``uint64``, singly occupied sites) and their complex amplitudes.  Spin
    bit ``s`` of the ``2**N`` spin basis is the spin of site ``s``.
    """
    check_shape(lx, ly)
    layers = circuit.schedule() if layers is None else layers
    n = lx * ly
    ids = np.arange(1 << n, dtype=np.int64)
    bits = [(ids >> s) & 1 for s in range(n)]
    v = None
    for layer in layers:
        if layer.stage != "spin":
            continue
        t = layer.angle
        if layer.kind == "neel":
            # site spinor (|0> - s|1>)/sqrt2 with s = stagger(site)
            v = np.ones(1 << n, complex) / math.sqrt(1 << n)
            for s in range(n):
                if stagger(s, lx) == 1:
                    v *= 1 - 2 * bits[s]
        elif layer.kind == "spin_x":
            for s in range(n):
                ii = ids[bits[s] == 0]
                jj = ii ^ (1 << s)
                x, y = v[ii].copy(), v[jj].copy()
                z = -1j * math.sin(t) * stagger(s, lx)
                v[ii] = math.cos(t) * x + z * y
                v[jj] = math.cos(t) * y + z * x
        elif layer.kind in ("spin_zz", "spin_yy"):
            for a, b in spin_bonds(lx, ly):
                zz = (1 - 2 * bits[a]) * (1 - 2 * bits[b])
                if layer.kind == "spin_zz":
                    v = v * np.exp(-1j * t * zz)
                else:  # exp(-i t Y_a Y_b)
                    v = math.cos(t) * v + 1j * math.sin(t) * zz * v[ids ^ (1 << a) ^ (1 << b)]
        elif layer.kind == "inject":
            occ = np.zeros(1 << n, np.uint64)
            for s in range(n):
                occ |= np.uint64(1) << (2 * s + bits[s]).astype(np.uint64)
            return occ, v
    raise ValueError("schedule has no injection layer")


def evolve(backend, lx: int, ly: int, bx: int, by: int, layers=None,
           callback=None):
    """Apply the fermionic part of the schedule to ``backend``.

    ``callback(stage_name)`` is invoked after the injection and at the end of
    each stage ('weak', 'pulse1', 'pulse2').
    """
    layers = circuit.schedule() if layers is None else layers
    edges = plaquette_edges(lx, ly)
    occ, amp = initial_spin_state(lx, ly, layers)
    backend.load(occ, amp)
    if callback:
        callback("injection")
    fermionic = [l for l in layers if l.stage != "spin"]
    for i, layer in enumerate(fermionic):
        if layer.kind == "hopping":
            for e in edges:
                if e.round == layer.round:
                    backend.hop(e.a, e.b, circuit.edge_amplitude(e, bx, by, layer))
        elif layer.kind == "onsite":
            backend.onsite(layer.angle)
        elif layer.kind == "spin_mixing":
            for s in range(lx * ly):
                backend.hop(2 * s, 2 * s + 1, circuit.spin_mixing_amplitude(s, lx, layer))
        else:
            raise ValueError(f"unexpected layer {layer.kind}")
        last = i + 1 == len(fermionic) or fermionic[i + 1].stage != layer.stage
        if callback and last:
            callback(layer.stage)
    return backend


class CPUBackend:
    """NumPy fixed-N backend; vector over all ``C(2N, N)`` occupation masks."""

    def __init__(self, lx: int, ly: int):
        check_shape(lx, ly)
        self.lx, self.ly, self.n = lx, ly, lx * ly
        m = 2 * self.n
        if m > 26:
            raise ValueError("CPU backend limited to 13 sites; use exact_gpu for 16 sites")
        allmasks = np.arange(1 << m, dtype=np.uint64)
        self.basis = allmasks[_popcount(allmasks) == self.n]
        self.dim = len(self.basis)
        self.v = np.zeros(self.dim, complex)
        pairs = (self.basis[:, None] >> (2 * np.arange(self.n, dtype=np.uint64))) & np.uint64(3)
        self._zz = np.sum(np.where((pairs == 1) | (pairs == 2), -1, 1), axis=1)
        self._doublons = np.sum(pairs == 3, axis=1)
        self._hop_cache = {}

    def index(self, masks):
        return np.searchsorted(self.basis, masks)

    def load(self, occupations, amplitudes):
        self.v[:] = 0
        self.v[self.index(np.asarray(occupations, np.uint64))] = amplitudes

    def _hop_tables(self, a, b):
        key = (a, b)
        if key not in self._hop_cache:
            ma, mb = np.uint64(1 << a), np.uint64(1 << b)
            between = np.uint64((1 << max(a, b)) - (1 << (min(a, b) + 1)))
            sel = np.flatnonzero(((self.basis & mb) != 0) & ((self.basis & ma) == 0))
            partner = self.index(self.basis[sel] ^ ma ^ mb)
            sign = 1 - 2 * _parity(self.basis[sel] & between).astype(float)
            self._hop_cache[key] = (sel, partner, sign)
        return self._hop_cache[key]

    def hop(self, a, b, z):
        """``exp[-i (z c_a^dag c_b + h.c.)]``."""
        mag = abs(z)
        if mag == 0:
            return
        i, j, sg = self._hop_tables(a, b)
        f = math.sin(mag) * z / mag
        x, y = self.v[i], self.v[j]
        self.v[i] = math.cos(mag) * x - 1j * np.conj(f) * sg * y
        self.v[j] = math.cos(mag) * y - 1j * f * sg * x

    def onsite(self, theta):
        """``D(theta) = exp[-i theta sum (n_up - 1/2)(n_down - 1/2)]``."""
        if theta:
            self.v *= np.exp(-0.25j * theta * self._zz)

    def measure(self):
        prob = np.abs(self.v) ** 2
        norm = float(prob.sum())
        doublons = float(prob @ self._doublons) / norm
        # eta |psi> on the N-2 sector, eta = sum_s stagger(s) c_s,down c_s,up
        keys, vals = [], []
        for s in range(self.n):
            mask = np.uint64(3 << (2 * s))
            sel = np.flatnonzero((self.basis & mask) == mask)
            keys.append(self.basis[sel] ^ mask)
            vals.append(stagger(s, self.lx) * self.v[sel])
        keys, vals = np.concatenate(keys), np.concatenate(vals)
        _, inverse = np.unique(keys, return_inverse=True)
        lowered = (np.bincount(inverse, vals.real) + 1j * np.bincount(inverse, vals.imag))
        eta2 = float(np.vdot(lowered, lowered).real) / norm
        return dict(norm=norm, doublons=doublons, eta_norm2=eta2,
                    A=circuit.pairing_from_moments(doublons, eta2, self.n))


def simulate(lx: int, ly: int, bx: int = 1, by: int = 1, backend=None,
             checkpoints: bool = False) -> dict:
    """Run the full two-step schedule in one boundary sector.

    Returns the measurement dict (``A``, ``doublons``, ``eta_norm2``,
    ``norm``) plus ``seconds``; with ``checkpoints=True`` also the
    measurements after each stage under ``'stages'``.
    """
    start = time.perf_counter()
    backend = CPUBackend(lx, ly) if backend is None else backend
    stages = {}
    cb = (lambda name: stages.__setitem__(name, backend.measure())) if checkpoints else None
    evolve(backend, lx, ly, bx, by, callback=cb)
    out = backend.measure()
    out.update(lattice=[lx, ly], boundary=[bx, by], seconds=time.perf_counter() - start)
    if checkpoints:
        out["stages"] = stages
    return out


def simulate_sectors(lx: int, ly: int, make_backend=None) -> dict:
    """All four boundary sectors and their uniform mean."""
    rows = {}
    for name, (bx, by) in BOUNDARIES.items():
        backend = make_backend(lx, ly) if make_backend else None
        rows[name] = simulate(lx, ly, bx, by, backend)
        rows[name].pop("stages", None)
    mean = sum(r["A"] for r in rows.values()) / 4
    return dict(sectors=rows, mean=mean)
