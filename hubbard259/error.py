"""Conditional error envelope for the 6x6 hierarchy estimate.

The finite connected-cumulant hierarchy is evaluated at particle rank r and
connected preparation-support cutoff c.  At r = 6 every cumulant needed by the
back-propagated observable is retained, so the remaining physical omission is
the contribution of connected preparation supports with seven or more sites
(the "support tail").  No rigorous bound on that tail is known.  This module
implements the empirical, pre-declared envelope used for the reported number.

Definitions (all quantities are per boundary sector b in {PP, AP, PA, AA}):

``d45_b = A_b(r6,c5) - A_b(r6,c4)`` and ``d56_b = A_b(r6,c6) - A_b(r6,c5)``
    the last two support increments of the 6x6 ladder.

``R``
    the calibration residual: the largest |A(r6,c6) - A_exact| over the eight
    small-lattice sectors where exact values exist (4x2 and 4x4, four
    boundaries each), using the same arithmetic as the 6x6 production.

``B_b = max(R, |d45_b|, |d56_b|)``
    the per-sector tail proxy.

Hypothesis H(tau): in every sector the omitted physical support tail is at
most ``tau * B_b``.  The reported envelope uses the pre-declared tau = 4.

Combination.  The reported estimate is the uniform mean of the four sector
values, A = (1/4) sum_b A_b(r6,c6).  Under H(tau) the triangle inequality
gives a physical allowance on the mean of ``(1/4) sum_b tau * B_b =
tau * mean(B_b)``.  The sector allowances are treated as fully correlated
systematic terms: they are averaged, never combined in quadrature, so there
is no 1/sqrt(4) reduction.  Two further terms are added linearly:

* ``E_prune``: the mean coefficient-deletion bound, exactly zero here because
  production deletes no coefficients;
* ``E_num = 0.002``: an empirical allowance for floating-point error of the
  mixed-precision arithmetic, qualified by the lower-cutoff comparisons and
  imaginary-part checks summarised in :data:`NUMERICAL_ALLOWANCE_EVIDENCE`.

Hence ::

    E_tot(tau) = tau * mean_b(B_b) + mean_b(E_prune,b) + E_num .

For the stored data every |d45_b|, |d56_b| is below R, so B_b = R for all four
sectors, mean(B) = R = 7.425943456630946e-4, and

    E_tot(4) = 4 * 7.425943456630946e-4 + 0 + 0.002 = 0.004970377382652378 .

This is a conditional empirical envelope, not a rigorous bound and not a
statistical confidence interval: H(4) (the transfer of the small-lattice
calibration to the 6x6 lattice) and the transfer of E_num to unsampled
cutoff-6 regions are assumptions.

The functions below take plain Python containers (as loaded from the JSON
files in ``results/``) and use only the standard library, with
:func:`math.fsum` for all sums so that results are reproducible bit for bit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SECTORS = ("PP", "AP", "PA", "AA")
TAU_REPORTED = 4
TAUS = (1, 2, 4)
NUMERICAL_ALLOWANCE = 0.002
TOLERANCE = 0.02
CALIBRATION_LATTICES = ("4x2", "4x4")

HYPOTHESIS = ("H(tau): in each boundary sector b the omitted physical preparation-support tail "
              "(connected supports of seven or more sites) is at most tau * B_b, with "
              "B_b = max(R, |d45_b|, |d56_b|); R is the worst |A(r6,c6) - exact| over the eight "
              "4x2/4x4 calibration sectors. tau = 4 was fixed before the 6x6 cutoff-6 values existed.")

#: Evidence for the empirical numerical allowance E_num (reported by the
#: precision study of the mixed-precision production; the underlying per-region
#: comparison data are not part of this repository).
NUMERICAL_ALLOWANCE_EVIDENCE = {
    "lower_cutoff_comparisons": 24,
    "lower_cutoff_reference": "float64 evaluation of the same hierarchy at cutoffs 4 and 5, all four sectors",
    "max_weighted_abs_regional_discrepancy": 4.29651e-7,
    "max_fraction_of_model_limit": 0.0190129,
    "max_complex_total_difference": 4.50965e-8,
    "cutoff6_imaginary_checks": 12,
    "max_cutoff6_imaginary_total": 4.36616e-8,
    "max_sampled_proxy_6x6_sector": 0.00018307828665,
    "scope": ("Observed comparisons that qualify E_num = 0.002 as an empirical allowance; they do not "
              "certify floating-point error on unsampled cutoff-6 regions."),
}

#: Weak-preparation parameter stress on 4x2 (lambda scales the weak spin and
#: hopping preparation angles only).  B uses the unchanged calibration R and the
#: perturbed d45/d56.  Reported values of a separate stress run; the raw
#: per-sector data are not included in this repository.
PARAMETER_STRESS_4X2 = (
    # lambda, exact mean, r6c6 mean, signed mean error, worst-sector |error|, max |error|/B
    (1.00, 0.408424424631482, 0.408718790382253, +0.000294365751, 0.000742508256, 0.115726),
    (1.25, 0.438967452789636, 0.440288908816033, +0.001321456026, 0.002703581098, 0.139342),
    (1.50, 0.453940945027257, 0.458203643481029, +0.004262698454, 0.006758077798, 0.154094),
    (2.00, 0.439609210872970, 0.461781687468749, +0.022172476596, 0.031234523399, 0.172637),
)


def mean(values) -> float:
    values = list(values)
    return math.fsum(values) / len(values)


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _index(records, lattice=None):
    out = {}
    for r in records:
        lat = r.get("lattice", lattice)
        key = (lat, r["boundary"], r["rank"], r["cutoff"], r.get("arithmetic", "mixed"))
        if key in out:
            raise ValueError(f"duplicate record {key}")
        out[key] = r
    return out


def support_ladder(records, rank=6, lattice="6x6", arithmetic="mixed"):
    """Per-sector A at cutoffs 4, 5, 6 and the increments d45, d56."""
    idx = _index(records, lattice)
    rows = []
    for b in SECTORS:
        a4, a5, a6 = (idx[(lattice, b, rank, c, arithmetic)]["A"] for c in (4, 5, 6))
        rows.append(dict(boundary=b, A_c4=a4, A_c5=a5, A_c6=a6, d45=a5 - a4, d56=a6 - a5))
    return rows


def calibration_table(small_records, rank=6, cutoff=6, lattices=CALIBRATION_LATTICES):
    """The eight matched calibration residuals A(r6,c6) - exact (same arithmetic as 6x6)."""
    idx = _index(small_records)
    rows = []
    for lat in lattices:
        for b in SECTORS:
            r = idx[(lat, b, rank, cutoff, "mixed")]
            residual = r["A"] - r["exact"]
            if abs(residual - r["residual"]) > 1e-15:
                raise ValueError(f"stored residual inconsistent for {lat} {b}")
            rows.append(dict(lattice=lat, boundary=b, A=r["A"], exact=r["exact"],
                             residual=residual, abs_residual=abs(residual)))
    return rows


def calibration_R(table) -> tuple[float, dict]:
    worst = max(table, key=lambda r: r["abs_residual"])
    return worst["abs_residual"], worst


def sector_proxies(ladder, R):
    return [dict(row, B=max(R, abs(row["d45"]), abs(row["d56"]))) for row in ladder]


def envelope(proxies, tau=TAU_REPORTED, numerical=NUMERICAL_ALLOWANCE, pruning=None):
    """E_tot(tau) = tau*mean(B) + mean(pruning) + E_num (see module docstring)."""
    mean_B = mean(p["B"] for p in proxies)
    mean_prune = mean(pruning) if pruning is not None else 0.0
    physical = tau * mean_B
    return dict(tau=tau, mean_B=mean_B, physical_allowance=physical, pruning_allowance=mean_prune,
                numerical_allowance=numerical, total=physical + (mean_prune + numerical))


def tau_max(proxies, numerical=NUMERICAL_ALLOWANCE, pruning_mean=0.0, tolerance=TOLERANCE):
    """Transfer factor at which E_tot would reach the tolerance (arithmetic only)."""
    return (tolerance - (pruning_mean + numerical)) / mean(p["B"] for p in proxies)


def holdout_checks(small_records, taus=TAUS):
    """Leave-one-lattice-out checks frozen before the cutoff-6 values existed.

    Training lattice T, held-out lattice H.  The training R is the largest
    |A(r6,c5) - exact| on T (float64 reference rows); for each held-out sector
    B = max(R_T, |d45_H|), with d45 from the same frozen rows.  The scored
    quantity is the matched direct cutoff-6 residual on H.
    """
    idx = _index(small_records)
    folds = []
    for train, held in (("4x2", "4x4"), ("4x4", "4x2")):
        R = max(abs(idx[(train, b, 6, 5, "fp64")]["residual"]) for b in SECTORS)
        rows = []
        for b in SECTORS:
            d45 = idx[(held, b, 6, 5, "fp64")]["A"] - idx[(held, b, 6, 4, "fp64")]["A"]
            B = max(R, abs(d45))
            res = idx[(held, b, 6, 6, "mixed")]["residual"]
            rows.append(dict(boundary=b, B=B, residual_r6c6=res,
                             passed={str(t): abs(res) <= t * B for t in taus},
                             ratio=abs(res) / B))
        folds.append(dict(training_lattice=train, heldout_lattice=held, training_R=R, rows=rows))
    return folds


def validation_8x2(small_records, R):
    """Apply the H(4) proxy, with the calibration R, to the independent 8x2 lattice."""
    ladder = support_ladder(small_records, rank=6, lattice="8x2", arithmetic="fp64")
    idx = _index(small_records)
    rows = []
    for row in sector_proxies(ladder, R):
        exact = idx[("8x2", row["boundary"], 6, 6, "fp64")]["exact"]
        err = row["A_c6"] - exact
        rows.append(dict(row, exact=exact, error_c6=err, ratio=abs(err) / row["B"],
                         passed_tau4=abs(err) <= TAU_REPORTED * row["B"]))
    return rows


def error_budget(h66: dict, small: dict) -> dict:
    """Full error budget from the stored 6x6 and small-lattice result files."""
    ladder = support_ladder(h66["records"])
    table = calibration_table(small["records"])
    R, worst = calibration_R(table)
    proxies = sector_proxies(ladder, R)
    pruning = [0.0 for _ in SECTORS]  # production deletes no coefficients
    env = {str(t): envelope(proxies, t, pruning=pruning) for t in TAUS}
    A = mean(p["A_c6"] for p in proxies)
    reported = env[str(TAU_REPORTED)]
    return dict(
        description=("Conditional error budget for the 6x6 rank-6 cutoff-6 estimate. "
                     "E_tot = tau*mean(B) + mean(E_prune) + E_num; reported tau = 4."),
        hypothesis=HYPOTHESIS,
        conditional=True, rigorous=False, statistical_coverage="none",
        estimate=dict(rank=6, cutoff=6, A=A, sectors={p["boundary"]: p["A_c6"] for p in proxies}),
        calibration=dict(R=R, worst_lattice=worst["lattice"], worst_boundary=worst["boundary"],
                         lattices=list(CALIBRATION_LATTICES)),
        sectors=[dict(boundary=p["boundary"], A=p["A_c6"], d45=p["d45"], d56=p["d56"],
                      latest_increment_grows=abs(p["d56"]) > abs(p["d45"]),
                      B=p["B"], tail_allowance_tau4=TAU_REPORTED * p["B"], pruning_bound=0.0)
                 for p in proxies],
        mean_increments=dict(d45=mean(p["d45"] for p in proxies), d56=mean(p["d56"] for p in proxies)),
        envelopes=env,
        reported_envelope=reported["total"],
        tau_max_for_tolerance=dict(tolerance=TOLERANCE, tau=tau_max(proxies)),
        numerical_allowance=dict(value=NUMERICAL_ALLOWANCE, evidence=NUMERICAL_ALLOWANCE_EVIDENCE),
        holdout_checks=holdout_checks(small["records"]),
        validation_8x2=validation_8x2(small["records"], R),
        parameter_stress_4x2=dict(
            description=("Reported 4x2 stress test: lambda scales only the weak preparation spin and hopping "
                         "angles. B uses the unchanged calibration R. Shows that accuracy is parameter "
                         "dependent; at lambda = 1 (the actual circuit) errors are far below 0.02."),
            columns=["lambda", "exact_mean", "A_mean_r6c6", "signed_mean_error",
                     "worst_sector_abs_error", "max_abs_error_over_B"],
            rows=[list(r) for r in PARAMETER_STRESS_4X2]),
    )


def calibration_report(small: dict) -> dict:
    table = calibration_table(small["records"])
    R, worst = calibration_R(table)
    return dict(
        description=("Matched calibration of the rank-6 cutoff-6 hierarchy against exact simulation on 4x2 and "
                     "4x4 (same mixed-precision arithmetic as the 6x6 production). R = max |residual|. "
                     "The 8x2 rows and hold-out checks are validation only and do not enter R."),
        rows=table, R=R, worst=dict(lattice=worst["lattice"], boundary=worst["boundary"]),
        holdout_checks=holdout_checks(small["records"]),
        validation_8x2=validation_8x2(small["records"], R),
    )


def results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results"
