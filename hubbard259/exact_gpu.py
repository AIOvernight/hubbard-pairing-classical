"""GPU backend (CuPy RawKernels) for the exact fixed-N simulator.

Same interface and conventions as :class:`hubbard259.exact.CPUBackend`; use
it through :func:`hubbard259.exact.simulate`::

    from hubbard259.exact import simulate
    from hubbard259.exact_gpu import GPUBackend
    r = simulate(4, 4, 1, 1, backend=GPUBackend(4, 4))

The state lives on the half-filled sector of up to 32 modes (16 sites, e.g.
4x4 or 8x2: ``C(32,16) = 601,080,390`` complex128 amplitudes, 9.6 GB) with
an explicit uint32 basis (2.4 GB).  Basis states are ranked with the
combinatorial number system using two 16-bit lookup tables.  The measurement
builds ``eta |psi>`` on the ``N-2`` sector (7.5 GB for 16 sites), so about
20 GB of device memory are needed for 16-site lattices.
"""
from __future__ import annotations

import math

import numpy as np

from . import circuit
from .lattice import check_shape

__all__ = ["GPUBackend", "rank_tables"]

KERNELS = r"""
#include <cuComplex.h>
typedef unsigned int U;
extern "C" {
__device__ U rankof(U s, const U* lo, const U* hi) { return lo[s & 65535] + hi[s >> 16]; }

__global__ void basisgen(U* basis, unsigned long long dim, int n, int m, const U* choose) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  U r = i, s = 0; int k = n;
  for (int p = m - 1; p >= 0 && k; p--) { U c = choose[p * 33 + k]; if (r >= c) { s |= 1u << p; r -= c; k--; } }
  basis[i] = s;
}

// exp[-i (z c_a^dag c_b + h.c.)]; (zr, zi) = sin|z| z/|z|, co = cos|z|.
__global__ void hop(cuDoubleComplex* v, const U* basis, unsigned long long dim, const U* lo, const U* hi,
                    U ma, U mb, U between, double co, double zr, double zi) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  U s = basis[i];
  if ((s & ma) || !(s & mb)) return;
  U j = rankof(s ^ ma ^ mb, lo, hi);
  double sg = (__popc(s & between) & 1) ? -1 : 1;
  cuDoubleComplex x = v[i], y = v[j];
  cuDoubleComplex f = make_cuDoubleComplex(zr * sg, zi * sg);
  cuDoubleComplex a = cuCmul(make_cuDoubleComplex(0, -1), cuCmul(cuConj(f), y));
  cuDoubleComplex b = cuCmul(make_cuDoubleComplex(0, -1), cuCmul(f, x));
  v[i] = cuCadd(make_cuDoubleComplex(co * x.x, co * x.y), a);
  v[j] = cuCadd(make_cuDoubleComplex(co * y.x, co * y.y), b);
}

__global__ void onsite(cuDoubleComplex* v, const U* basis, unsigned long long dim, int sites, double theta) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  U s = basis[i]; int zz = 0;
  for (int j = 0; j < sites; j++) zz += (((s >> (2 * j)) ^ (s >> (2 * j + 1))) & 1) ? -1 : 1;
  double t = -theta * zz / 4;
  v[i] = cuCmul(v[i], make_cuDoubleComplex(cos(t), sin(t)));
}

// out = eta |psi> on the N-2 sector (gather form).
__global__ void eta(cuDoubleComplex* out, unsigned long long dim, int sites, int lx, const U* choose,
                    const U* lo, const U* hi, const cuDoubleComplex* v) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  U r = i, s = 0; int k = sites - 2;
  for (int p = 2 * sites - 1; p >= 0 && k; p--) { U c = choose[p * 33 + k]; if (r >= c) { s |= 1u << p; r -= c; k--; } }
  cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
  for (int j = 0; j < sites; j++) {
    U mask = 3u << (2 * j);
    if (!(s & mask)) {
      cuDoubleComplex a = v[rankof(s | mask, lo, hi)];
      double sign = ((j % lx + j / lx) & 1) ? -1 : 1;
      sum.x += sign * a.x; sum.y += sign * a.y;
    }
  }
  out[i] = sum;
}

// per-block partial sums of norm, doublons
__global__ void moments(double* out, const cuDoubleComplex* v, const U* basis, unsigned long long dim, int sites) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  double norm = 0, dbl = 0;
  if (i < dim) {
    U s = basis[i]; cuDoubleComplex z = v[i]; norm = z.x * z.x + z.y * z.y; int d = 0;
    for (int j = 0; j < sites; j++) d += ((s >> (2 * j)) & 3) == 3;
    dbl = norm * d;
  }
  __shared__ double sh[2][256]; int t = threadIdx.x; sh[0][t] = norm; sh[1][t] = dbl; __syncthreads();
  for (int q = 128; q; q /= 2) { if (t < q) { sh[0][t] += sh[0][t + q]; sh[1][t] += sh[1][t + q]; } __syncthreads(); }
  if (!t) { out[2 * blockIdx.x] = sh[0][0]; out[2 * blockIdx.x + 1] = sh[1][0]; }
}

__global__ void normblocks(double* out, const cuDoubleComplex* v, unsigned long long dim) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  double z = 0; if (i < dim) { cuDoubleComplex a = v[i]; z = a.x * a.x + a.y * a.y; }
  __shared__ double sh[256]; int t = threadIdx.x; sh[t] = z; __syncthreads();
  for (int q = 128; q; q /= 2) { if (t < q) sh[t] += sh[t + q]; __syncthreads(); }
  if (!t) out[blockIdx.x] = sh[0];
}
}
"""


def rank_tables(n: int):
    """Binomial table and 16-bit split ranking tables for ``n`` particles."""
    choose = np.zeros((33, 33), np.uint32)
    for p in range(33):
        for k in range(p + 1):
            choose[p, k] = math.comb(p, k)
    low = np.zeros(65536, np.uint32)
    high = np.zeros(65536, np.uint32)
    for s in range(65536):
        bits = [p for p in range(16) if s >> p & 1]
        low[s] = sum(math.comb(p, k + 1) for k, p in enumerate(bits))
        nl = n - len(bits)
        if 0 <= nl <= 16:
            high[s] = sum(math.comb(p + 16, nl + k + 1) for k, p in enumerate(bits))
    return choose, low, high


class GPUBackend:
    """CuPy fixed-N backend (up to 16 sites)."""

    threads = 256

    def __init__(self, lx: int, ly: int):
        import cupy as cp
        check_shape(lx, ly)
        self.cp = cp
        self.lx, self.ly, self.n = lx, ly, lx * ly
        if 2 * self.n > 32:
            raise ValueError("GPU backend supports at most 32 modes (16 sites)")
        self.dim = math.comb(2 * self.n, self.n)
        self.module = cp.RawModule(code=KERNELS, options=("--std=c++11",))
        choose, low, high = rank_tables(self.n)
        self._low, self._high = low, high
        self.choose, self.lo, self.hi = cp.asarray(choose), cp.asarray(low), cp.asarray(high)
        self.basis = cp.empty(self.dim, cp.uint32)
        self.v = cp.zeros(self.dim, cp.complex128)
        self._launch("basisgen", self.dim, self.basis, np.uint64(self.dim), np.int32(self.n),
                     np.int32(2 * self.n), self.choose)

    def _launch(self, name, dim, *args):
        blocks = (dim + self.threads - 1) // self.threads
        self.module.get_function(name)((blocks,), (self.threads,), args)

    def load(self, occupations, amplitudes):
        occ = np.asarray(occupations, np.uint64).astype(np.uint32)
        ranks = self._low[occ & 65535].astype(np.int64) + self._high[occ >> 16]
        self.v[:] = 0
        self.v[self.cp.asarray(ranks)] = self.cp.asarray(np.asarray(amplitudes, complex))

    def hop(self, a, b, z):
        mag = abs(z)
        if mag == 0:
            return
        f = math.sin(mag) * z / mag
        between = (1 << max(a, b)) - (1 << (min(a, b) + 1))
        self._launch("hop", self.dim, self.v, self.basis, np.uint64(self.dim), self.lo, self.hi,
                     np.uint32(1 << a), np.uint32(1 << b), np.uint32(between),
                     np.float64(math.cos(mag)), np.float64(f.real), np.float64(f.imag))

    def onsite(self, theta):
        if theta:
            self._launch("onsite", self.dim, self.v, self.basis, np.uint64(self.dim),
                         np.int32(self.n), np.float64(theta))

    def measure(self):
        cp = self.cp
        blocks = (self.dim + self.threads - 1) // self.threads
        part = cp.empty((blocks, 2), cp.float64)
        self._launch("moments", self.dim, part, self.v, self.basis, np.uint64(self.dim), np.int32(self.n))
        norm, dbl = (float(x) for x in cp.sum(part, axis=0).get())
        del part
        dim2 = math.comb(2 * self.n, self.n - 2)
        lowered = cp.empty(dim2, cp.complex128)
        self._launch("eta", dim2, lowered, np.uint64(dim2), np.int32(self.n), np.int32(self.lx),
                     self.choose, self.lo, self.hi, self.v)
        nb = cp.empty((dim2 + self.threads - 1) // self.threads, cp.float64)
        self._launch("normblocks", dim2, nb, lowered, np.uint64(dim2))
        eta2 = float(cp.sum(nb).get())
        del lowered, nb
        cp.get_default_memory_pool().free_all_blocks()
        doublons, eta2 = dbl / norm, eta2 / norm
        return dict(norm=norm, doublons=doublons, eta_norm2=eta2,
                    A=circuit.pairing_from_moments(doublons, eta2, self.n))

    def state(self):
        """Host copy of the state vector (in basis order of :attr:`basis`)."""
        return self.v.get()
