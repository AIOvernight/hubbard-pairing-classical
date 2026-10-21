"""Tests for the exact fixed-N simulator."""
import itertools

import numpy as np
import pytest
from scipy import sparse

from hubbard259 import circuit
from hubbard259.exact import CPUBackend, evolve, simulate

REFERENCE_4X2 = {(1, 1): 0.48510449121154775, (-1, 1): 0.49078793651909564,
                 (1, -1): 0.3131753546070669, (-1, -1): 0.3446299161882184}


@pytest.mark.parametrize("bx,by", list(REFERENCE_4X2))
def test_4x2_reference_values(bx, by):
    r = simulate(4, 2, bx, by)
    assert abs(r["norm"] - 1) < 1e-12
    assert abs(r["A"] - REFERENCE_4X2[(bx, by)]) < 1e-12


class DenseBackend:
    """Independent full-Fock reference built from Jordan-Wigner matrices."""

    def __init__(self, nsites):
        self.n, self.m = nsites, 2 * nsites
        dim = 1 << self.m
        states = np.arange(dim)
        self.c = []
        for mode in range(self.m):
            occ = states[(states >> mode) & 1 == 1]
            sign = [(-1) ** bin(s & ((1 << mode) - 1)).count("1") for s in occ]
            self.c.append(sparse.csr_matrix((sign, (occ ^ (1 << mode), occ)), shape=(dim, dim)))
        self.num = [((states >> mode) & 1).astype(float) for mode in range(self.m)]
        self.interaction = sum((self.num[2 * s] - .5) * (self.num[2 * s + 1] - .5) for s in range(nsites))
        self.v = np.zeros(dim, complex)

    def load(self, occ, amp):
        self.v[:] = 0
        self.v[np.asarray(occ, np.int64)] = amp

    def hop(self, a, b, z):
        # H = z c_a^dag c_b + h.c. satisfies H^2 = |z|^2 P with P the projector
        # on states where exactly one of a, b is occupied, hence
        # exp(-iH) = 1 - P + cos|z| P - i sin|z| H/|z|.
        h = z * (self.c[a].T @ self.c[b])
        h = h + h.conj().T
        r = abs(z)
        p = self.num[a] + self.num[b] - 2 * self.num[a] * self.num[b]
        self.v = self.v - p * self.v + np.cos(r) * p * self.v - 1j * np.sin(r) / r * (h @ self.v)

    def onsite(self, theta):
        self.v = np.exp(-1j * theta * self.interaction) * self.v


def test_cpu_backend_matches_dense_jordan_wigner_2x2():
    for bx, by in itertools.product((1, -1), repeat=2):
        dense, fast = DenseBackend(4), CPUBackend(2, 2)
        evolve(dense, 2, 2, bx, by)
        evolve(fast, 2, 2, bx, by)
        full = np.zeros_like(dense.v)
        full[fast.basis.astype(np.int64)] = fast.v
        assert np.linalg.norm(full - dense.v) < 1e-12
        # observable from explicit operators
        n = 4
        dop = sum(dense.num[2 * s] * dense.num[2 * s + 1] for s in range(n))
        eta = sum((-1) ** ((s % 2) + (s // 2)) * (dense.c[2 * s + 1] @ dense.c[2 * s]) for s in range(n))
        v = dense.v
        a = circuit.pairing_from_moments(np.vdot(v, dop * v).real, np.linalg.norm(eta @ v) ** 2, n)
        assert abs(a - fast.measure()["A"]) < 1e-12


def test_gpu_backend_matches_cpu_4x2():
    pytest.importorskip("cupy")
    from hubbard259.exact_gpu import GPUBackend
    try:
        gpu = GPUBackend(4, 2)
    except Exception as exc:  # no device / driver
        pytest.skip(f"GPU unavailable: {exc}")
    cpu = CPUBackend(4, 2)
    evolve(cpu, 4, 2, -1, 1)
    evolve(gpu, 4, 2, -1, 1)
    assert np.linalg.norm(cpu.v - gpu.state()) < 1e-12
    assert abs(gpu.measure()["A"] - REFERENCE_4X2[(-1, 1)]) < 1e-12
