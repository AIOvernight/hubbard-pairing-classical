"""Classical reference calculations for the two-step Hubbard pairing circuit.

Submodules
----------
lattice    geometry, site-major mode ordering, boundary sectors
circuit    gate schedule, parameters and observable conventions
exact      exact fixed-particle-number state-vector simulator (CPU)
exact_gpu  the same simulator on a CUDA GPU via CuPy (optional)
error      error envelope from stored results

Connected-cumulant hierarchy (overview in ``hubbard259.hierarchy``)
hierarchy   algorithm overview; exports ``run_hierarchy``
reduction   observable pulled back through G2 and D(3): dressed operator on <=4 sites
clusters    exact local preparation of connected clusters, support census, incidence weights
cumulants   local raw moments and fermionic connected cumulants
projection  projection of cumulants through the first Gaussian layer (CPU and CUDA)
readout     readout-region orbits, moment reconstruction and contraction
production  driver evaluating all (rank, cutoff) models for one lattice sector
"""
__version__ = "1.0.0"
