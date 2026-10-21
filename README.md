# Classical estimate for the two-step 6×6 Hubbard pairing circuit

This repository is a classical response to the **two-step case** of
[Quantum Advantage Tracker issue #259](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/259)
("Superconducting pairing correlations"): a 6×6 Hubbard model at half filling with U/t = 8 and
mixed boundary conditions. The circuit prepares a low-energy state and then applies a light pulse made of two
Trotter steps of size 0.375/t. The device ran it on 90 qubits (72 system qubits and 18 ancillas of a local
fermionic encoding), taking 400 shots.

It gives a classical estimate of the noiseless value of the measured pairing statistic, the code that computes
it, exact simulators for smaller lattices used as controls, the stored results, and the arithmetic of the error
estimate. **It does not address the three-step or four-step circuits.** The four-step circuit is the headline
target of the issue.

## Result

| Boundary sector (b_x, b_y) | A |
|---|---:|
| PP (+1, +1) | 0.3832431205 |
| AP (−1, +1) | 0.3836285668 |
| PA (+1, −1) | 0.3836290019 |
| AA (−1, −1) | 0.3840705515 |
| **Equal-weight mean (mixed boundary conditions)** | **0.3836428102** |

**A = 0.38364 ± 0.00497 (conditional envelope; see [Error estimate](#error-estimate)).**

The four sectors are computed separately at the same setting. AP and PA are related by a lattice symmetry, but
the approximation does not use that symmetry to infer one sector from the other. In the ideal circuit the mixed
boundary preparation gives each sector weight exactly 1/4. The value is the uncorrected output of the finite
hierarchy described below: no experimental number is used to fit, select or correct it.

## Observable convention

The positive number reported for the experiment (0.266 for two steps) is the statistic

```
A = (2/N) [ <D> − <η† η> ],     N = 36,
D = Σ_j n_{j↑} n_{j↓},          η = Σ_j (−1)^{x_j + y_j} c_{j↓} c_{j↑}.
```

The authors' analysis computes this statistic from the measured shots. With the paper's pair correlator P_η(x,y)
(its Eq. (6), which includes the Hermitian conjugate), A is equivalently −Σ_{(x,y)≠0} (−1)^{x+y} P_η(x,y).
Taken literally, the paper's displayed sign convention gives **P = −A**. The staggered sum written in the issue
has neither the Hermitian conjugate nor the overall minus sign. In a state of fixed total particle number it
therefore equals **−A/2**. Compare numbers only within one convention:

| Convention | This work (classical) | Experiment (2 steps, 400 shots) |
|---|---:|---:|
| A = (2/N)[⟨D⟩ − ⟨η†η⟩] (reported positive statistic) | 0.3836428 | 0.2658 ± 0.0255 |
| Paper's displayed P = −A | −0.3836428 | −0.2658 ± 0.0255 |
| Literal issue expression = −A/2 | −0.1918214 | −0.1329 ± 0.0127 |

The authors' per-distance CSV files can be summed with the weight −(−1)^{x+y}. This gives ≈ 0.265, consistent
with the shot-level estimator (0.2658). The CSV files pool pairs per displacement, which is why the two numbers
are slightly different. A ≠ −A/2 and A ≠ −A are not sign corrections to apply to the experiment. The table
only lists the same quantity in three notations.

## Method

The calculation never builds the 72-mode many-body state. The prepared state is evaluated through local
connected cumulants, and the strong part of the circuit is moved into the observable.

1. **Circuit.** After the preparation, whose last layer is the first half-step D(1.5) and is treated as part of the prepared state, the remaining pulse is `D(1.5) G2 D(3) G1` (operators act right to left). Here `D(θ)` is the onsite
   interaction layer `exp[−iθ Σ_j (n_{j↑}−½)(n_{j↓}−½)]`. `G1` and `G2` are the two Gaussian (free-fermion)
   hopping layers, with time step 0.375 and diagonal-field coefficients (k, j) = (−0.7511, −0.47413) and
   (+0.7511, −0.47413). The preparation uses the specified rounded parameters, starting from the X-Néel product
   state. It is not replaced by an exact ground state or a thermal state. The schedule was checked against the
   authors' circuit generator in two ways. An exact 20-qubit simulation of the encoded 4×2 circuit agrees with
   the fermionic simulation of our schedule to 6×10⁻¹⁵. The released 90-qubit two-step circuit has the same
   ideal output distribution as the circuit regenerated from the authors' code. On 4×2 the vertical bonds
   are doubled, so this comparison cannot tell apart some orientation and sign conventions. Each of those
   conventions was checked to be an exact symmetry of the circuit for every lattice size. For larger lattices
   the link to the literal circuit is therefore indirect; there is no direct encoded comparison at 4×4 or 6×6.
   Those two checks are not part of this repository. What this repository does include is
   `scripts/validate_circuit.py`. It parses the released two-step Guppy file statically and maps all 1404
   non-Clifford gates onto the 16 layers of `hubbard259.circuit`. For each gate it checks the angle
   magnitude, the layer order, the gate counts and the qubit pairs. It does not check rotation signs.
2. **Exact cancellation of the last onsite layer.** `D` and the total particle number commute with every pair
   transfer η_i† η_j. As a result, the final `D(1.5)` drops out of ⟨A⟩ exactly.
3. **Back-propagation of the observable.** A Gaussian unitary maps annihilators to linear combinations of
   annihilators, so `G2† A G2` is exactly quartic. Conjugating it by the central layer `D(3)` multiplies each of
   the four acted-on modes by at most one opposite-spin number factor. The dressed observable is therefore a sum
   of terms of **particle rank ≤ 6 on at most four sites** (eight modes), written in the modes dressed by
   `G1`. This is an exact algebraic statement. The strong pulses are not truncated or expanded in a small
   parameter. Every one of the 7423 translation orbits of ≤4-site readout regions on the 6×6 torus (66711
   regions) is included.
4. **Connected preparation cumulants.** Take each connected set T of sites. The local state on T is prepared
   exactly with all one-site gates, and the weak two-site preparation gates (spin-preparation ZZ/YY and
   preparation hopping) are switched on only inside T. Fermionic embedding signs and the boundary sign of each
   sector are included, and six-site winding supports are prepared separately for each boundary. Local
   reduced-density-matrix cumulants are formed, and the contributions of all proper connected subclusters are
   subtracted. Disconnected supports contribute nothing, because cumulants are additive over components with
   fixed particle number. Summing the connected contributions over **all supports of up to c = 6 sites**
   (10992 supports on 6×6) approximates the cumulants of the prepared state.
5. **Projection, then summation, then moment reconstruction.** Cumulants transform under the one-particle map
   of `G1` exactly like moments. The local cumulants can therefore be projected into each readout region. All
   projected cumulants are **summed first**. Only then are moments up to rank r = 6 reconstructed and
   contracted with the dressed observable. Reconstructing moments support by support and adding scalar results
   would drop the disconnected products, which would be a different approximation.

The only physical approximation is the cutoff on preparation supports. Connected contributions from seven or
more sites are omitted. At r = 6 there is no separate readout-rank truncation. The reconstructed truncated
moments need not come from a positive density matrix, so no norm- or fidelity-based error argument applies.
Production arithmetic is mixed precision. Local cumulants, compounds, inversion and weighted sums are in
complex128. The projections of rank-three cumulants and of raw ranks four to six are in complex64.

This setting is called r6c6 below, meaning reconstruction rank 6 and support cutoff 6. The stored results
contain all nine combinations r ∈ {4, 5, 6} × c ∈ {4, 5, 6} for the four sectors.

The cumulant machinery (generating-function cumulants of reduced density matrices) is standard; see
W. Kutzelnigg and D. Mukherjee, *Chem. Phys. Lett.* (1999), "Direct determination of the cumulants of the
reduced density matrices". What this repository adds is the reduction specific to this circuit, and its use as
a finite hierarchy for it.

## Error estimate

**There is no proven bound on the omitted preparation supports.** The quoted ±0.0049704 is a *conditional
envelope*. It holds if an empirical transfer hypothesis holds, and it is neither a confidence interval nor a
certified bound.

**How it is built.** Let R = 7.426×10⁻⁴ be the worst absolute r6c6 error over the eight exactly solvable
small-system sectors (4×2 and 4×4, four boundaries each). It is computed in the production arithmetic, and the
worst case is the 4×2 AP sector. For each 6×6 sector b, let d45_b and d56_b be the last two support-cutoff
increments at r = 6. The hypothesis **H(4)** states that the omitted tail of each sector is at most 4·B_b, with
B_b = max(R, |d45_b|, |d56_b|). The factor 4 was fixed before the 6×6 cutoff-six values were computed. All
6×6 increments (5.0×10⁻⁴ to 5.9×10⁻⁴) are below R, so B_b = R in every sector and the physical part is 4R =
0.0029704. A further **numerical allowance of 0.002** for the mixed-precision arithmetic is added in full,
which gives **0.0029704 + 0.002 = 0.0049704**. The sector allowances are treated as fully correlated, with no
√4 reduction.

**What supports the hypothesis.** Controls at the specified circuit parameters:

- **4×2 (all four sectors).** Exact simulation is available, and at c = 8 the hierarchy is complete. The
  r6c6 errors are −5.0×10⁻⁵ (PP), +7.43×10⁻⁴ (AP), +2.0×10⁻⁴ (PA) and +2.85×10⁻⁴ (AA). The worst of these
  sets R.
- **4×4 (all four sectors).** The exact mean is 0.3844142, reproduced by two independently written
  fixed-particle-number simulators to 2×10⁻¹⁶. The r6c6 hierarchy errs by at most 1.9×10⁻⁴ per sector, with a
  mean error of +1.30×10⁻⁴.
- **8×2 (all four sectors).** The exact mean is 0.4073249. A double-precision evaluation of the same hierarchy
  at r6c6 errs by at most 9.2×10⁻⁵ per sector. This lattice is not part of the calibration of R. It is an
  extra validation control, added after the hypothesis was fixed, and its largest error/B ratio is 0.0695, far
  inside H(4).
- **Leave-one-geometry-out tests.** There are two held-out folds: calibrate on 4×2 and test on 4×4, then the
  reverse. Each fold uses only lower-cutoff information from its training lattice. Both folds pass in all four
  sectors at factors 1, 2 and 4, with a largest residual/proxy ratio of 0.116. Their proxies are
  2.66–8.98 times the final target calibration R, so these tests do not validate coverage directly at
  the final target threshold.

These lattices have 8 or 16 sites, and their supports wrap around circumferences of at most eight sites. 6×6
is the first geometry with large numbers of open connected supports of seven or more sites. The small-system
agreement is therefore evidence for transferring the hypothesis to 6×6, not proof that it transfers.

**Evidence that limits the claim.**

- *Stress test.* Multiplying all weak preparation angles by λ = 2 on 4×2 produces a mean r6c6 error of 0.022
  (0.031 in the worst sector). The cutoff increments grow at the same time, so H(4) still covers it. The
  method is not uniformly accurate across parameters. At λ = 1.25 and λ = 1.5 the worst-sector errors are
  2.7×10⁻³ and 6.8×10⁻³. Only the summary rows of this test are stored, in `results/error_budget.json`.
- *No clear convergence on 6×6.* The last two increments are both ≈5×10⁻⁴ and positive in every sector
  (mean d45 = +5.48×10⁻⁴, mean d56 = +5.33×10⁻⁴). In PP the last increment is **13% larger** than the
  previous one. Each net increment is also a cancellation residue: the sum of absolute regional changes is 3–4
  times larger than the net change. This is why no convergence- or extrapolation-based bound is claimed. On
  the small lattices, the higher-cutoff controls are not monotone either.
- *Scalar cross-check.* A separate scalar resummation at support cutoff four gives 0.3858288423, which is
  +0.00219 from the r6c6 mean and inside the envelope. It omits the same supports, so the agreement tests the
  implementation, not the truncation, and it is not used to shrink the envelope. The scalar route is not part of
  this repository; its value is quoted as reported.
- *Numerical allowance.* Every observed numerical discrepancy is far below 0.002, including:
  - complete c4/c5 comparisons against a separate calculation in other arithmetic (total differences ≤ 4.51×10⁻⁸);
  - selected c6 regions recomputed in full double precision (closures within 3.7×10⁻⁹);
  - imaginary parts of the sector totals ≤ 4.4×10⁻⁸.

  The 0.002 is still an empirical allowance, not a floating-point certificate for the regions that were not
  sampled.
- *Partial independent recomputation.* An independent reimplementation of the reduction and hierarchy
  reproduces all of the following to ~10⁻¹⁴ relative to a double-precision evaluation of the same hierarchy (the stored mixed-precision values agree with it to ≤ 1.3×10⁻⁷):
  - the complete 4×2 hierarchy through c6;
  - all four 4×4 sectors at c4;
  - the complete PP sector of 6×6 at c4;
  - selected 6×6 contractions at c6.

  The full 6×6 c6 production itself was computed once and not independently recomputed.

An error below 0.005, or even below 0.02, is **not established unconditionally**. The quoted envelope
requires H(4) and the numerical allowance.

## Comparison with the experiment

Re-parsing the authors' released shots ([Zenodo record 17483197](https://doi.org/10.5281/zenodo.17483197))
reproduces the reported two-step value, **A = 0.2658 ± 0.0255**. The uncertainty is the sample standard error
over 400 shots. The authors' estimator rescales the usable system-site pairs of each shot and then averages
over shots. Only 48 of the 400 shots have all 36 sites usable. Relating this estimator to the ideal
unconditioned expectation value therefore requires assumptions about leakage and missingness. The classical
estimate is higher (0.3836 versus 0.2658). Because the systematic and selection effects of the estimator are not
quantified, **we do not claim a statistical incompatibility, and we do not attribute the difference to any
specific noise mechanism.** A noiseless reference for the full 6×6 two-step value from the authors (for
example, an emulator value) would test this estimate directly.

The authors' circuits and data are **not redistributed** here. The scripts download them from the issue
attachments and from Zenodo.

## Reproducing

Requirements: Python ≥ 3.10, NumPy, SciPy, Numba, pytest and matplotlib (`requirements.txt`). The GPU paths
additionally need CuPy (`requirements-gpu.txt`), CUDA 12.x and an NVIDIA GPU.
The download helper also uses Bash, curl and sha256sum. Development used Python 3.12, NumPy 2.5,
SciPy 1.18, Numba 0.68 and CuPy 14.2 on an RTX 5090 (32 GB).

```sh
pip install -r requirements.txt            # CPU
pip install -r requirements-gpu.txt        # optional, GPU
```

`reproduce.sh` has three tiers:

| Command | What it does | Hardware | Measured time |
|---|---|---|---|
| `./reproduce.sh verify` | Unit tests. Then `verify.py` (hashes in `results/SHA256SUMS`, means, increments and envelope recomputed from the stored results) and `scripts/estimate_error.py --check`. | CPU | 18 s in a clean environment; `verify.py` alone takes 0.01 s |
| `./reproduce.sh cheap` | Exact 4×2 (CPU) and exact 4×4 (GPU) simulations, all four sectors. One 6×6 boundary sector of the hierarchy at cutoff c = 4 (GPU). Each result is compared with `results/`. | CPU + GPU (≈20 GB for 4×4) | Exact components: 0.8 s CPU + 18.2 s GPU; complete PP c4 hierarchy: 101.7 s GPU |
| `./reproduce.sh full` | Full 6×6 production: all four boundaries, all r/c settings through r6c6. Builds a raw-factor cache on disk (≈1.3 GB). | GPU + disk | Original computation: 115.4 min (RTX 5090); this implementation not yet timed end to end |

The clean-environment verification used `OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=8`. It passed
32 tests and skipped the optional external-circuit check. When a GPU is unavailable, the optional
CPU/GPU state-comparison test also skips; the tier can run entirely on CPU.

The individual steps can also be run directly:

```sh
python verify.py                                   # stored results: hashes, means, envelope, CSV/JSON (CPU, < 1 s)
bash scripts/fetch_circuits.sh                     # downloads the authors' Guppy circuits into data/circuits (SHA256-checked)
python scripts/validate_circuit.py data/circuits/superconducting_pairing_2steps.py   # schedule check (CPU, ~2 s)
python scripts/run_exact.py --lattice 4x2                  # exact 4x2, all sectors (CPU, 0.8 s)
python scripts/run_exact.py --lattice 4x4 --backend gpu    # exact 4x4 (GPU, ~20 GB, 18 s)
python scripts/run_exact.py --lattice 8x2 --backend gpu    # exact 8x2 (GPU, ~20 GB, 20 s)
python scripts/run_hierarchy.py --geometry 4x2 --boundary PP --rank 4,5,6 --cutoff 4,5,6 \
    --device cpu --out runs/h_4x2_PP.json          # hierarchy on 4x2 (CPU)
python scripts/run_hierarchy.py --geometry 6x6 --boundary PP --rank 4,5,6 --cutoff 4 \
    --device gpu --out runs/h_6x6_PP_c4.json       # one 6x6 sector at cutoff 4 (GPU)
python scripts/convention_check.py                 # downloads the Zenodo shots into data/zenodo (MD5-checked),
                                                   # reproduces A = 0.2658 +/- 0.0255 and the converted values (~10 s)
python scripts/estimate_error.py --check           # recomputes calibration and the conditional envelope 0.0049704
python scripts/make_figures.py                     # regenerates results/figures/
python scripts/hash_results.py --check             # checks results/SHA256SUMS
```

Measured on this code: the exact 4×2 simulation takes 0.8 s on CPU. The exact 4×4 and 8×2 simulations take
18 s and 20 s on an RTX 5090 for all four sectors (C(32,16) ≈ 6×10⁸ amplitudes per sector). All of them match
the stored values to ≤ 2×10⁻¹⁶. The original full 6×6
production (36 models: four sectors × nine settings, with 7423 readout orbits per sector) took 115.4 minutes on
the GPU. That is a historical measurement of the computation that generated the stored results, not an
end-to-end timing of this cleaned implementation.

## Repository layout

```
hubbard259/                 Python package
  lattice.py                6x6 / 4x4 / 4x2 / 8x2 torus geometry, boundary signs, checkerboard plaquettes
  circuit.py                gate schedule and parameters of the two-step circuit
  exact.py, exact_gpu.py    exact fixed-particle-number simulators (CPU / CuPy)
  reduction.py              back-propagated observable (rank <= 6 on <= 4 sites)
  clusters.py               connected preparation supports and subcluster weights
  cumulants.py              local reduced-density-matrix cumulants and moment reconstruction
  projection.py             projection of cumulants through the first Gaussian layer
  readout.py                readout-region orbits and contraction with the dressed observable
  production.py             6x6 production driver (all sectors and r/c settings)
  error.py                  conditional H(4) envelope from stored results
scripts/                    command-line entry points (see Reproducing)
tests/                      pytest suite (CPU, fast)
results/                    stored outputs
  hierarchy_6x6.{json,csv}  4 sectors x 9 (r, c) settings
  exact_small.{json,csv}    exact 4x2, 4x4 and 8x2 values, all sectors
  calibration.{json,csv}    small-system hierarchy versus exact
  error_budget.json         inputs and result of the conditional envelope
  SHA256SUMS                checksums of the stored results
  figures/                  convergence and calibration plots
verify.py                   fast CPU consistency check of results/
reproduce.sh                verify / cheap / full tiers
docs/STATUS.md              status of the result
docs/AI_AND_PROVENANCE.md   how the work was produced; which checks are independent
MANIFEST.json               SHA-256 and size of every file in the repository
LICENSE, LICENSE-DATA.md    MIT (code), CC BY 4.0 (data and documentation)
```

## Author

**AIOvernight** ([@AIOvernight](https://x.com/AIOvernight) on X).

This work was produced by AI agents; see [docs/AI_AND_PROVENANCE.md](docs/AI_AND_PROVENANCE.md) for what that means for the
checks, and [docs/STATUS.md](docs/STATUS.md) for the status of the result.

## Citation

If you use this code or these numbers, please cite this repository (see `CITATION.cff`) together with the
original experiment and its data:

- Superconducting pairing correlations on a trapped-ion quantum computer, [arXiv:2511.02125](https://arxiv.org/abs/2511.02125).
- E. Granet, S.-H. Lin, K. Hémery, M. Iqbal, H. Dreyer, *Supporting Data for the publication "Superconducting
  pairing correlations on a trapped-ion quantum computer"*, Zenodo (2025),
  [doi:10.5281/zenodo.17483197](https://doi.org/10.5281/zenodo.17483197).

```bibtex
@software{aiovernight_hubbard259_2026,
  author  = {{@AIOvernight}},
  title   = {Classical estimate of the two-step 6x6 Hubbard pairing circuit (Quantum Advantage Tracker issue 259)},
  version = {1.0.0},
  year    = {2026},
  month   = {10},
  url     = {https://x.com/AIOvernight},
  license = {MIT}
}
```

## License

Code: MIT (`LICENSE`). Data, tables and documentation: CC BY 4.0 (`LICENSE-DATA.md`).

Project initiation and coordination: **[@AIOvernight](https://x.com/AIOvernight)**.

*A night full of tokens + RTX 5090*
