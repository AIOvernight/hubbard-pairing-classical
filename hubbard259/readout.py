"""Readout regions, moment reconstruction (closure) and observable contraction.

Every dressed monomial lives on the site support of its quartic, a region of
1..4 sites.  The total is a sum over regions; exact even translations map
regions to orbits whose multiplicities weight a single canonical
representative (6x6: 66711 supports in 7423 orbits).

For each region the summed projected cumulants K_1..K_r (rank cutoff r) are
turned into moments by the bottom-up recursion of :mod:`hubbard259.cumulants`
(``Gamma(empty)=1``; cumulants above rank r are set to zero), and the region
terms of the dressed operator are contracted with those moments.
"""
import itertools

import numpy as np

from .cumulants import forward_cumulants
from .reduction import dressed_terms, region_terms


def canonical_region(region, lx, ly):
    return min(tuple(sorted((p % lx + dx) % lx + lx * ((p // lx + dy) % ly) for p in region))
               for dx in range(0, lx, 2) for dy in range(0, ly, 2))


def region_orbits(lx, ly, size, orbits=True):
    """Canonical regions of ``size`` sites and their orbit multiplicities."""
    groups = {}
    for region in itertools.combinations(range(lx * ly), size):
        key = canonical_region(region, lx, ly) if orbits else region
        groups[key] = groups.get(key, 0) + 1
    return list(groups), list(groups.values())


def measure_region_cpu(kappas, region, rank, lx, ly, bx, by):
    """CPU closure + contraction for one region from summed cumulants (complex128)."""
    terms = region_terms(lx, ly, bx, by, region)
    if not terms:
        return 0j
    m = 2 * len(region)
    tables, looks = forward_cumulants(kappas, m, rank)
    total = 0j
    for (a, b), weight in terms.items():
        r = a.bit_count()
        total += weight * (1 if r == 0 else tables[r - 1][looks[r - 1][a], looks[r - 1][b]])
    return total


_MOMENTS_SOURCE = r'''
struct Z { double re,im;
 __device__ Z(double a=0.,double b=0.):re(a),im(b){}
 __device__ Z operator+(Z b)const{return Z(re+b.re,im+b.im);}
 __device__ Z operator-()const{return Z(-re,-im);}
 __device__ Z operator*(Z b)const{return Z(re*b.re-im*b.im,re*b.im+im*b.re);}
 __device__ Z& operator+=(Z b){re+=b.re;im+=b.im;return *this;}
};
extern "C" __global__ void moments(const Z* kap,Z* gamma,const int* dims,const int* offsets,const int* look,const int* masks,int rank,int Q,int R,int M,int size){
 int n=dims[rank];long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(id>=(long long)R*n*n)return;
 int j=id%n;long long t=id/n;int i=t%n;int batch=t/n;
 int am=masks[i],bm=masks[j],low=am&-am,rest=am^low;Z value=0.;
 int extra=rest;
 while(true){
  int a=low|extra,k=__popc((unsigned)a);
  if(k<=Q){
   int asign=0;
   for(int p=0;p<M;p++)if(a&(1<<p))asign+=__popc((unsigned)(am&((1<<p)-1)));
   asign-=k*(k-1)/2;
   int b=bm;
   while(b){
    if(__popc((unsigned)b)==k){
     int sign=asign;
     for(int p=0;p<M;p++)if(b&(1<<p))sign+=__popc((unsigned)(bm&((1<<p)-1)));
     sign-=k*(k-1)/2;
     int ia=look[k*(1<<M)+a],ib=look[k*(1<<M)+b];
     int l=rank-k,ja=look[l*(1<<M)+(am^a)],jb=look[l*(1<<M)+(bm^b)];
     Z term=kap[batch*size+offsets[k]+ia*dims[k]+ib]*gamma[batch*size+offsets[l]+ja*dims[l]+jb];
     value+=(sign&1)?-term:term;
    }
    b=(b-1)&bm;
   }
  }
  if(extra==0)break;extra=(extra-1)&rest;
 }
 gamma[batch*size+offsets[rank]+i*n+j]=value;
}
'''


class MomentGPU:
    """Batched closure (cumulants truncated at rank ``rank``) and contraction, complex128."""

    def __init__(self, s, rank):
        import cupy as cp
        from .projection import MomentLayout
        self.cp = cp
        self.s = s
        self.q = rank
        self.layout = lay = MomentLayout(2 * s)
        self.kernel = cp.RawKernel(_MOMENTS_SOURCE, 'moments', options=('--std=c++17',))
        quartets = [q for q in itertools.product(range(s), repeat=4) if len(set(q)) == s]
        inds, weights, qids = [], [], []
        for i, q in enumerate(quartets):
            for (am, bm), v in dressed_terms(*q).items():
                r = int(am).bit_count()
                inds.append(int(lay.offset_cpu[r]) + int(lay.look_cpu[r, am]) * lay.dims_cpu[r] + int(lay.look_cpu[r, bm]))
                weights.append(v)
                qids.append(i)
        self.quartets = cp.asarray(np.array(quartets, np.int32))
        self.indices = cp.asarray(np.array(inds, np.int32))
        self.weights = cp.asarray(np.array(weights))
        self.qids = cp.asarray(np.array(qids, np.int32))

    def moments(self, kappas):
        cp = self.cp
        lay = self.layout
        R = len(kappas[0])
        kap = cp.zeros((R, lay.size), cp.complex128)
        gamma = cp.zeros_like(kap)
        gamma[:, 0] = 1
        for r, k in enumerate(kappas, 1):
            kap[:, int(lay.offset_cpu[r]):int(lay.offset_cpu[r + 1])] = k.reshape(R, -1)
        for r in range(1, lay.nr + 1):
            n = lay.dims_cpu[r]
            size = R * n * n
            self.kernel(((size + 127) // 128,), (128,),
                        (kap, gamma, lay.dims, lay.offsets, lay.look, lay.masks[r - 1], np.int32(r), np.int32(self.q),
                         np.int32(R), np.int32(lay.m), np.int32(lay.size)))
        return gamma

    def contract(self, gamma, regions, coeff):
        """Sum over all quartets with support exactly the region (coeff on device)."""
        cp = self.cp
        regions = cp.asarray(np.asarray(regions, np.int32))
        q = regions[:, self.quartets]
        cc = coeff[q[:, :, 0], q[:, :, 1], q[:, :, 2], q[:, :, 3]]
        return (gamma[:, self.indices] * self.weights[None, :] * cc[:, self.qids]).sum(axis=1)
