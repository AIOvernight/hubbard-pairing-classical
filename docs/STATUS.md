# Status

| Item | Status |
|---|---|
| Result type | Classical numerical estimate of the noiseless two-step pairing statistic A for the 6×6 circuit of Quantum Advantage Tracker issue #259 |
| Value | A = 0.3836428102 (equal-weight mean of four boundary sectors, rank 6, preparation-support cutoff 6) |
| Error | ±0.0049704 conditional envelope: empirical hypothesis H(4) (physical tail) + 0.002 empirical numerical allowance. **Not a proven bound, not a confidence interval.** An error ≤ 0.02 is not established unconditionally. |
| Verified scope | Exact controls on 4×2, 4×4 (calibration) and 8×2 (validation); held-out cross-geometry tests; numerical comparisons listed in the README. 6×6 cutoff-six production computed once; not independently recomputed in full. |
| Scope of the claim | Two-step circuit only. The three- and four-step circuits are not addressed. |
| Comparison with experiment | Reproduced hardware value 0.2658 ± 0.0255 (same convention). No claim of incompatibility or of a cause of the difference. |
| Human review | None. See [AI_AND_PROVENANCE.md](AI_AND_PROVENANCE.md). |
| Peer review | None. The title describes the computation; it does not claim priority or peer review. |
| Reproducibility | `./reproduce.sh verify` (CPU, seconds); `./reproduce.sh cheap` (minutes, GPU); full cutoff-six production ≈ 2 GPU-hours. |
| Licence | Code: MIT (`LICENSE`). Data, tables and documentation: CC BY 4.0 (`LICENSE-DATA.md`). |
| Open items | A proven bound on preparation supports of seven or more sites; a full independent recomputation of the 6×6 cutoff-six production. |
