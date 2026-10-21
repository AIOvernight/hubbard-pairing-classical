"""Tests for the conditional error envelope and the stored-result verifier."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hubbard259 import error  # noqa: E402

RESULTS = ROOT / "results"


@pytest.fixture(scope="module")
def data():
    return error.load_json(RESULTS / "hierarchy_6x6.json"), error.load_json(RESULTS / "hierarchy_small.json")


def test_reported_envelope_exact(data):
    budget = error.error_budget(*data)
    assert budget["calibration"]["R"] == 0.0007425943456630946
    assert (budget["calibration"]["worst_lattice"], budget["calibration"]["worst_boundary"]) == ("4x2", "AP")
    assert all(s["B"] == budget["calibration"]["R"] for s in budget["sectors"])
    assert budget["reported_envelope"] == 0.004970377382652378
    assert f"{budget['reported_envelope']:.7f}" == "0.0049704"
    assert budget["envelopes"]["1"]["total"] == 0.0027425943456630946
    assert budget["envelopes"]["2"]["total"] == 0.003485188691326189
    assert budget["tau_max_for_tolerance"]["tau"] == 24.239344273389296
    assert budget["estimate"]["A"] == 0.3836428101992484


def test_increments(data):
    ladder = {r["boundary"]: r for r in error.support_ladder(data[0]["records"])}
    assert ladder["PP"]["d45"] == pytest.approx(0.000500404122282, abs=1e-15)
    assert ladder["PP"]["d56"] == pytest.approx(0.000565699215731, abs=1e-15)
    assert abs(ladder["PP"]["d56"]) > abs(ladder["PP"]["d45"])  # PP latest increment grows by 13%
    assert ladder["AA"]["d56"] == pytest.approx(0.000500975678723, abs=1e-15)


def test_proxy_uses_increment_when_larger():
    ladder = [dict(boundary=b, A_c4=0.0, A_c5=0.01, A_c6=0.015, d45=0.01, d56=0.005) for b in error.SECTORS]
    env = error.envelope(error.sector_proxies(ladder, 0.001), tau=4)
    assert env["mean_B"] == 0.01
    assert env["total"] == pytest.approx(4 * 0.01 + 0.002, abs=0)


def test_no_quadrature_reduction():
    proxies = [dict(B=b) for b in (1e-3, 2e-3, 3e-3, 4e-3)]
    env = error.envelope(proxies, tau=4, numerical=0.0)
    assert env["total"] == pytest.approx(4 * 2.5e-3, rel=1e-15)


def test_holdout_and_8x2(data):
    folds = error.holdout_checks(data[1]["records"])
    assert [f["training_R"] for f in folds] == [0.004203334475645304, 0.001964626719730578]
    assert all(all(r["passed"].values()) for f in folds for r in f["rows"])
    assert max(r["ratio"] for f in folds for r in f["rows"]) == pytest.approx(0.11574, abs=1e-5)
    rows = error.validation_8x2(data[1]["records"], 0.0007425943456630946)
    assert all(r["passed_tau4"] for r in rows)
    assert max(r["ratio"] for r in rows) == pytest.approx(0.069541, abs=1e-6)


def test_stored_budget_matches(data):
    stored = error.load_json(RESULTS / "error_budget.json")
    assert json.loads(json.dumps(error.error_budget(*data))) == stored


def _run_verify(results):
    return subprocess.run([sys.executable, "-B", str(ROOT / "verify.py"), "--results", str(results)],
                          capture_output=True, text=True, timeout=60)


def test_verify_passes():
    proc = _run_verify(RESULTS)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS:" in proc.stdout


def test_verify_detects_tampering(tmp_path):
    copy = tmp_path / "results"
    shutil.copytree(RESULTS, copy)
    path = copy / "hierarchy_6x6.json"
    h = json.loads(path.read_text())
    h["records"][0]["A"] += 1e-12
    path.write_text(json.dumps(h, indent=1))
    proc = _run_verify(copy)
    assert proc.returncode == 1
    assert "FAIL" in proc.stdout
