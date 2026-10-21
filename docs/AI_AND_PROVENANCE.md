# AI use and provenance

## How this work was produced

The calculation, the code, the error analysis and this repository were produced by AI agents (models:
Claude by Anthropic and GPT by OpenAI), coordinated by [@AIOvernight](https://x.com/AIOvernight).
**No human or institutional review of the science or the code has taken place.** Nothing in this
repository is peer reviewed.

## What "independent" means here

Several checks below are independent **in implementation**: separate code written from the method
description, run on different hardware paths or arithmetic. They are **not** independent in
authorship: all of them were written by AI agents within the same project.

| Check | Independent in implementation from |
|---|---|
| Exact simulations: the CPU (`hubbard259/exact.py`) and GPU (`hubbard259/exact_gpu.py`) simulators agree on the full 4×2 state to machine precision; the GPU 4×4 and 8×2 values match earlier separately written exact simulations to ≤ 2×10⁻¹⁶ | each other; the hierarchy code |
| Dressed-observable reduction vs direct Fock evolution on 4×2 (~10⁻¹⁵) | the hierarchy contraction |
| Double-precision reimplementation of the reduction and hierarchy (4×2 through c6, 4×4 at c4, 6×6 PP at c4, selected 6×6 c6 contractions; ~10⁻¹⁴) | the mixed-precision production code |
| Scalar resummation at support cutoff four (0.3858288423) — not included in this repository | the cumulant route (it shares the omitted supports) |
| Recomputation of the error-envelope arithmetic from the stored tables (`verify.py`, `scripts/estimate_error.py`) | the original scoring code |
| Static check of the authors' released two-step circuit against the gate schedule (`scripts/validate_circuit.py`) | the simulators |
| Reproduction of the experimental value from the authors' public raw shots (`scripts/convention_check.py`) | — |
| Reproduction from a clean copy of this repository in an isolated environment (verify tier, exact 4×2/4×4, 6×6 PP sector at cutoff 4) | the environment in which the code was developed |

None of these checks makes the error envelope a proven bound; see the README, section *Error estimate*.

## What was preserved and what was changed when packaging

Preserved:
- The numerical method and arithmetic of the production hierarchy (rank ≤ 6, preparation supports
  ≤ 6 sites, the mixed-precision scheme). The packaged code reproduces the archived production totals of the 6×6 PP sector (ranks 4–6 at cutoff 4, all 7423 readout orbits) within the 1×10⁻⁸ tolerance checked by `./reproduce.sh cheap`; during packaging it was also compared against the archived 4×2 values (all sectors, all nine settings) and the 4×4 PP sector at cutoff 4.
- All numbers in `results/` are taken unchanged (full precision) from the original production and
  control records; `results/SHA256SUMS` fixes them.

Changed:
- Code was reorganised into the `hubbard259` package, renamed, documented and stripped of internal bookkeeping. The exact simulators,
  the circuit validator and the convention script are clean rewrites validated against the archived values.
- Results were converted to plain JSON/CSV tables; figures were regenerated from those tables.
- The authors' circuit files and raw data are not redistributed; scripts download them from their
  public locations and check their hashes.
- Only the method that produced the reported number is included.

Not re-run with the packaged code: the full 6×6 production at support cutoff six (about two GPU-hours
for the four sectors). The stored cutoff-six values come from the original production run of the same
arithmetic.
