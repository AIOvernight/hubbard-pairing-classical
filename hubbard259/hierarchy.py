"""Connected-cumulant hierarchy for the two-step pairing observable (overview).

Problem
-------
A 6x6 half-filled Hubbard preparation (72 modes, 36 fermions) is followed by
``G1 D(3) G2 D(1.5)`` and the pairing statistic
``A = (2/L)(<D> - <eta^+ eta^->)`` is measured, averaged over the PP, AP, PA
and AA boundary sectors.  No global state vector is ever formed.

Algorithm
---------
1. Observable reduction (:mod:`hubbard259.reduction`).  D(1.5) commutes with
   A and cancels.  Pulling A back through G2 and the central onsite layer D(3)
   gives, exactly, a dressed operator whose normal-ordered monomials have
   particle rank <= 6 and live on at most four sites (8 modes).  On 4x2 it
   agrees with direct Fock evolution to ~1e-15 (35,456 monomials).

2. Preparation cumulants (:mod:`hubbard259.clusters`,
   :mod:`hubbard259.cumulants`).  For every connected site support T with
   |T| <= c the preparation restricted to T (all local/fixed gates and every
   weak two-site gate inside T, fermionic embedding signs, spectators as
   fixed singly occupied spinors) is evolved exactly in local Fock space.
   Its raw moments Gamma_1..Gamma_6 give local cumulants; connected
   preparation cumulants follow by subtracting all connected proper
   subclusters, implemented through integer incidence weights.  Six-site
   supports share one canonical source per even-translation orbit
   (seam/gauge phases on the embedded modes).

3. Projection (:mod:`hubbard259.projection`).  Cumulants are covariant under
   the exact first Gaussian layer: K_r(S) = conj(C_r(W1)) kappa_r(T) C_r(W1)^T
   for each readout region S.  Weighted projected cumulants of all clusters
   are summed BEFORE the nonlinear moment reconstruction, so products of
   disconnected preparation contributions are restored by the closure.

4. Readout (:mod:`hubbard259.readout`).  Moments are rebuilt from summed
   cumulants truncated at rank r (r = 6 retains every cumulant this observable
   can see), the dressed operator is contracted region by region and the
   orbit-weighted regions are summed with ``math.fsum``.

Controls are the rank cutoff r in {4, 5, 6} and the connected support cutoff
c in {4, 5, 6}; at c the connected preparation contributions on more than c
sites are omitted.  Reconstructed moments need not define a positive state.

Arithmetic
----------
The stored 6x6 numbers use the ``mixed`` GPU policy: compounds,
cumulant inversion, weighted sums and closure in complex128; raw ranks 1-2 in
complex128; the exact local kappa_3 (complex128) projected in complex64 with
Gamma_3 reconstructed in complex128; raw ranks 4-6 projected in complex64;
cuBLAS pedantic math and TF32 disabled.  ``double`` (GPU) and the CPU path are
complex128 throughout; on small lattices they differ from ``mixed``
by up to ~1.3e-7 in A.

Entry point: :func:`run_hierarchy` (also ``scripts/run_hierarchy.py``).
"""
from .production import run_hierarchy  # noqa: F401

__all__ = ['run_hierarchy']
