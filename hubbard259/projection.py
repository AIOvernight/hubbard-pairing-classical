"""Projection of local preparation cumulants through the exact first Gaussian layer.

For a readout region S (modes of at most four sites) and a source cluster T,
let ``W_eff = W1[S_modes, T_modes] * phases`` and ``U_r = C_r(W_eff)`` its rank-r
compound matrix (all r x r minors, rows/columns in sorted bitmask order).
Raw moments transform as ``Gamma_r(S) = conj(U_r) Gamma_r(T) U_r^T``; since a
Gaussian map acts linearly on every rank, cumulants transform the same way:

    K_r(S) = conj(U_r) kappa_r(T) U_r^T .

The projected cumulants of all clusters are multiplied by their integer
incidence weights and summed BEFORE the nonlinear moment reconstruction in
:mod:`hubbard259.readout`.

Two implementations are provided:

* CPU, complex128 throughout (:func:`project_one`), used for small lattices
  and as an independent check;
* CUDA (:class:`FactorGPU`), batched over regions x clusters, with compounds
  built by an exact division-free Laplace recurrence that reuses rank r-1
  minors.  Arithmetic policy ``mixed`` (the production setting):

  - compounds, cumulant inversion and all reductions in complex128;
  - raw ranks 1, 2 projected in complex128;
  - rank 3: the exact local kappa_3 (complex128 on CPU) is projected in
    complex64 and the output Gamma_3 is reconstructed in complex128 as
    projected kappa_3 + lower-rank terms;
  - raw ranks 4..6 projected in complex64 (two GEMMs per rank).

  ``double`` keeps every operation in complex128.
"""
import itertools

import numpy as np
from numba import njit, prange

from .cumulants import comb_modes, entry_moments, inverse_moments, third_cumulant

PRECISIONS = ('double', 'mixed')


# ----------------------------------------------------------------------------
# CPU reference.
# ----------------------------------------------------------------------------

@njit(cache=True)
def determinant_small(m):
    d = 1. + 0j
    for j in range(len(m)):
        p = j
        for i in range(j + 1, len(m)):
            if abs(m[i, j]) > abs(m[p, j]):
                p = i
        if m[p, j] == 0:
            return 0. + 0j
        if p != j:
            for k in range(j, len(m)):
                m[j, k], m[p, k] = m[p, k], m[j, k]
            d = -d
        d *= m[j, j]
        for i in range(j + 1, len(m)):
            q = m[i, j] / m[j, j]
            for k in range(j + 1, len(m)):
                m[i, k] -= q * m[j, k]
    return d


@njit(parallel=True, cache=True)
def compound_rect(w, left, right):
    """Minors det w[left_i, right_j] for all index rows of ``left``/``right``."""
    k = left.shape[1]
    out = np.empty((len(left), len(right)), np.complex128)
    for i in prange(len(left)):
        for j in range(len(right)):
            sub = np.empty((k, k), np.complex128)
            for a in range(k):
                for b in range(k):
                    sub[a, b] = w[left[i, a], right[j, b]]
            out[i, j] = determinant_small(sub)
    return out


def project_one(record, w, output, order=6):
    """Projected cumulants K_1..K_cap of one source record onto output modes."""
    m = len(output)
    k = len(record['sites'])
    cap = min(order, m, 2 * k)
    ww = np.ascontiguousarray(w[np.ix_(output, record['modes'])] * np.asarray(record['phases'])[None, :])
    tables = [np.ones((1, 1), complex)]
    for r in range(1, cap + 1):
        entry = record['entries'][r - 1]
        a = comb_modes(m, r)
        if entry['kind'] == 'zero':
            tables.append(np.zeros((len(a), len(a)), complex))
            continue
        u = compound_rect(ww, a, comb_modes(2 * k, r))
        if entry['kind'] == 'gram':
            y = u @ entry['value']
            tables.append(np.ascontiguousarray(y.conj() @ y.T))
        else:
            tables.append(np.ascontiguousarray(u.conj() @ entry['value'] @ u.T))
    return inverse_moments(tables, m, cap)


def mask_basis(m, r):
    return np.array(sorted(sum(1 << p for p in s) for s in itertools.combinations(range(m), r)), np.int32)


# ----------------------------------------------------------------------------
# CUDA implementation.
# ----------------------------------------------------------------------------

_COMPLEX = r'''
struct Z { double re,im;
 __device__ Z(double a=0.,double b=0.):re(a),im(b){}
 __device__ Z operator+(Z b)const{return Z(re+b.re,im+b.im);}
 __device__ Z operator-(Z b)const{return Z(re-b.re,im-b.im);}
 __device__ Z operator-()const{return Z(-re,-im);}
 __device__ Z operator*(Z b)const{return Z(re*b.re-im*b.im,re*b.im+im*b.re);}
 __device__ Z& operator+=(Z b){re+=b.re;im+=b.im;return *this;}
};
'''

_LAPLACE_SOURCE = _COMPLEX + r'''
extern "C" __global__ void compounds_laplace(const Z* w,const int* outm,const int* inm,const double* phase,const int* left,const int* right,const int* lookleft,const int* lookright,const Z* previous,Z* current,int M,int R,int C,int A,int B,int oldA,int oldB,int mo,int mi){
 long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(id>=(long long)R*C*A*B)return;
 int b=id%B;long long t=id/B;int a=t%A;t/=A;int c=t%C;int q=t/C;
 int am=left[a],bm=right[b];int first=__ffs(am)-1;
 int pa=lookleft[am^(1<<first)];int remaining=bm;int position=0;Z value=0.;
 while(remaining){
  int bit=remaining&-remaining;int col=__ffs(bit)-1;int pb=lookright[bm^bit];
  Z x=w[outm[q*mo+first]*M+inm[c*mi+col]]*phase[c*mi+col];
  Z term=x*previous[((long long)q*C+c)*oldA*oldB+(long long)pa*oldB+pb];
  value+=(position&1)?-term:term;remaining^=bit;position++;
 }
 current[id]=value;
}
'''

# Moment -> cumulant inversion, one rank per launch.  In mixed mode the rank-3
# slot of ``gamma`` initially holds the projected kappa_3; it is kept as the
# cumulant and gamma_3 is reconstructed as kappa_3 + lower-rank terms.
_INVERSE_SOURCE = _COMPLEX + r'''
extern "C" __global__ void inverse(Z* gamma,Z* kap,const int* dims,const int* offsets,const int* look,const int* masks,int rank,int R,int M,int size,int kappa3_input){
 int n=dims[rank];long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(id>=(long long)R*n*n)return;
 int j=id%n;long long t=id/n;int i=t%n;int batch=t/n;
 int am=masks[i],bm=masks[j],low=am&-am,rest=am^low;Z value=0.;int extra=rest;
 while(true){
  int a=low|extra,k=__popc((unsigned)a);
  if(k<rank){
   int asign=-k*(k-1)/2,aa=a;
   while(aa){int bit=aa&-aa;asign+=__popc((unsigned)(am&(bit-1)));aa^=bit;}
   int b=bm;
   while(b){
    if(__popc((unsigned)b)==k){
     int sign=asign-k*(k-1)/2,bb=b;
     while(bb){int bit=bb&-bb;sign+=__popc((unsigned)(bm&(bit-1)));bb^=bit;}
     int ia=look[k*(1<<M)+a],ib=look[k*(1<<M)+b];int l=rank-k;
     int ja=look[l*(1<<M)+(am^a)],jb=look[l*(1<<M)+(bm^b)];
     Z term=kap[batch*size+offsets[k]+ia*dims[k]+ib]*gamma[batch*size+offsets[l]+ja*dims[l]+jb];
     value+=(sign&1)?-term:term;
    }b=(b-1)&bm;
   }
  }if(extra==0)break;extra=(extra-1)&rest;
 }
 int pos= batch*size+offsets[rank]+i*n+j;
 if(kappa3_input && rank==3){Z third=gamma[pos];gamma[pos]=third+value;kap[pos]=third;}
 else kap[pos]=gamma[pos]-value;
}
'''


def low_precision(precision, rank):
    """True when the raw projection of ``rank`` runs in complex64."""
    if precision not in PRECISIONS:
        raise ValueError(f'Unknown precision {precision!r}')
    return precision == 'mixed' and rank > 2


class MomentLayout:
    """Packed storage of ranks 0..min(6, m) for m local modes (device arrays)."""

    def __init__(self, m):
        import cupy as cp
        from .cumulants import basis
        self.m = m
        self.nr = min(6, m)
        dims = [1] + [len(basis(m, r)) for r in range(1, self.nr + 1)]
        offsets = np.array([0] + list(np.cumsum(np.array(dims) ** 2)), np.int32)
        self.dims = cp.asarray(np.array(dims, np.int32))
        self.offsets = cp.asarray(offsets)
        self.size = int(offsets[-1])
        look = np.zeros((self.nr + 1, 1 << m), np.int32)
        self.masks = []
        for r in range(1, self.nr + 1):
            masks = basis(m, r).astype(np.int32)
            look[r, masks] = np.arange(len(masks))
            self.masks.append(cp.asarray(masks))
        self.look = cp.asarray(look)
        self.look_cpu = look
        self.offset_cpu = offsets
        self.dims_cpu = dims


class LaplaceCompoundsGPU:
    """Sequential compound matrices of W_eff for regions x clusters, sharing minors.

    Call ``begin()``, then ``next(1)``, ``next(2)``, ...; returned arrays must
    not be modified while the next rank is being built.
    """

    def __init__(self, w, outm, inm, maxorder=6, right_phases=None):
        import cupy as cp
        self.cp = cp
        self.w = w
        self.outm = outm
        self.inm = inm
        self.R, self.mo = outm.shape
        self.C, self.mi = inm.shape
        self.phase = cp.ones((self.C, self.mi), cp.float64) if right_phases is None else cp.asarray(right_phases, dtype=cp.float64)
        self.kernel = cp.RawKernel(_LAPLACE_SOURCE, 'compounds_laplace', options=('--std=c++17',))
        self.data = []
        oldleft = np.array([0], np.int32)
        oldright = np.array([0], np.int32)
        for rank in range(1, min(maxorder, self.mo, self.mi) + 1):
            left = mask_basis(self.mo, rank)
            right = mask_basis(self.mi, rank)
            ll = np.full(1 << self.mo, -1, np.int32)
            lr = np.full(1 << self.mi, -1, np.int32)
            ll[oldleft] = np.arange(len(oldleft))
            lr[oldright] = np.arange(len(oldright))
            self.data.append((cp.asarray(left), cp.asarray(right), cp.asarray(ll), cp.asarray(lr), len(oldleft), len(oldright)))
            oldleft, oldright = left, right
        self.begin()

    def begin(self):
        self.previous = self.cp.ones((self.R, self.C, 1, 1), self.cp.complex128)
        self.rank = 0

    def next(self, rank):
        if rank != self.rank + 1 or rank > len(self.data):
            raise ValueError('Compound ranks must be sequential')
        cp = self.cp
        left, right, ll, lr, oldA, oldB = self.data[rank - 1]
        A, B = len(left), len(right)
        out = cp.empty((self.R, self.C, A, B), cp.complex128)
        self.kernel(((out.size + 255) // 256,), (256,),
                    (self.w, self.outm, self.inm, self.phase, left, right, ll, lr, self.previous, out,
                     np.int32(len(self.w)), np.int32(self.R), np.int32(self.C), np.int32(A), np.int32(B),
                     np.int32(oldA), np.int32(oldB), np.int32(self.mo), np.int32(self.mi)))
        self.previous = out
        self.rank = rank
        return out


class FactorGPU:
    """Batched projection of one group of equal-size source clusters.

    ``project()`` returns, for the current batch of equal-size regions, the
    projected cumulants ``[K_1, ..., K_cap]`` with shape (R, C, A_r, A_r).
    """

    def __init__(self, records, regions, w_device, layout, precision='mixed', order=6):
        import cupy as cp
        low_precision(precision, 1)
        self.cp = cp
        self.precision = precision
        self.records = records
        self.C = len(records)
        self.k = len(records[0]['sites'])
        self.m = 2 * len(regions[0])
        self.cap = min(order, self.m, 2 * self.k)
        self.layout = layout
        self.kernel = cp.RawKernel(_INVERSE_SOURCE, 'inverse', options=('--std=c++17',))
        self.inm = cp.asarray(np.array([r['modes'] for r in records], np.int32))
        phases = np.asarray([r['phases'] for r in records], np.float64)
        if phases.shape != (self.C, 2 * self.k) or not np.isin(phases, [-1, 1]).all():
            raise ValueError('Invalid source mode phases')
        self.w = w_device
        self.set_regions(regions, phases)
        self.entries = []
        for r in range(1, self.cap + 1):
            kind = records[0]['entries'][r - 1]['kind']
            if any(x['entries'][r - 1]['kind'] != kind for x in records):
                raise ValueError('Mixed factor kinds inside one group')
            if precision == 'mixed' and r == 3:
                # Also for two-site sources, where raw Gamma_3 = 0 but kappa_3 != 0.
                kind = 'dense'
                value = cp.asarray(np.array([third_cumulant(x) for x in records]), dtype=cp.complex64)
            elif kind == 'zero':
                value = None
            else:
                dtype = cp.complex64 if low_precision(precision, r) else cp.complex128
                value = cp.asarray(np.array([x['entries'][r - 1]['value'] for x in records]), dtype=dtype)
            self.entries.append((kind, value))

    def set_regions(self, regions, phases=None):
        cp = self.cp
        if any(2 * len(r) != self.m for r in regions):
            raise ValueError('Output support size cannot change')
        self.R = len(regions)
        outm = cp.asarray(np.array([[2 * s + t for s in r for t in (0, 1)] for r in regions], np.int32))
        if phases is not None:
            self.laplace = LaplaceCompoundsGPU(self.w, outm, self.inm, self.cap, right_phases=phases)
        else:
            self.laplace.outm = outm
            self.laplace.R = self.R

    def project(self):
        cp = self.cp
        lay = self.layout
        batch = self.R * self.C
        gamma = cp.zeros((batch, lay.size), cp.complex128)
        gamma[:, 0] = 1
        self.laplace.begin()
        for r, (kind, value) in enumerate(self.entries, 1):
            if kind == 'zero':
                continue  # all higher ranks are zero as well (rank > cluster particle number)
            u = self.laplace.next(r)
            if low_precision(self.precision, r):
                u = u.astype(cp.complex64)
            if kind == 'gram':
                y = u @ value[None, :, :, :]
                raw = y.conj() @ y.swapaxes(-1, -2)
            else:
                raw = (u.conj() @ value[None, :, :, :]) @ u.swapaxes(-1, -2)
            gamma[:, int(lay.offset_cpu[r]):int(lay.offset_cpu[r + 1])] = raw.reshape(batch, -1)
        kap = cp.zeros_like(gamma)
        for r in range(1, self.cap + 1):
            n = lay.dims_cpu[r]
            size = batch * n * n
            self.kernel(((size + 127) // 128,), (128,),
                        (gamma, kap, lay.dims, lay.offsets, lay.look, lay.masks[r - 1], np.int32(r), np.int32(batch),
                         np.int32(self.m), np.int32(lay.size), np.int32(self.precision == 'mixed')))
        self.laplace.previous = None
        return [kap[:, int(lay.offset_cpu[r]):int(lay.offset_cpu[r + 1])].reshape(self.R, self.C, lay.dims_cpu[r], lay.dims_cpu[r])
                for r in range(1, self.cap + 1)]
