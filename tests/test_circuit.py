"""Tests for lattice geometry, gate schedule and the circuit validator."""
import os
import pathlib
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from hubbard259 import circuit, lattice

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("lx,ly", [(4, 2), (4, 4), (6, 6), (8, 2)])
def test_each_bond_once_per_spin_and_step(lx, ly):
    edges = lattice.plaquette_edges(lx, ly)
    assert len(edges) == 2 * 2 * 4 * (lx // 2) * (ly // 2)
    bonds = Counter((e.spin, frozenset(e.sites)) for e in edges)
    nn = set()
    for s in range(lx * ly):
        x, y = lattice.coordinates(s, lx)
        nn.add(frozenset((s, lattice.site_index(x + 1, y, lx, ly))))
        nn.add(frozenset((s, lattice.site_index(x, y + 1, lx, ly))))
    for spin in (0, 1):
        got = {b for (sp, b) in bonds if sp == spin}
        if ly > 2 and lx > 2:
            assert got == nn and all(bonds[(spin, b)] == 1 for b in nn)
        else:
            assert got == nn


def test_seam_flags_and_boundary_names():
    for e in lattice.plaquette_edges(6, 6):
        (xa, ya), (xb, yb) = (lattice.coordinates(s, 6) for s in e.sites)
        assert e.wraps_x == ({xa, xb} == {0, 5} and ya == yb)
        assert e.wraps_y == ({ya, yb} == {0, 5} and xa == xb)
        assert e.a % 2 == e.b % 2
    for name, (bx, by) in lattice.BOUNDARIES.items():
        assert lattice.boundary_name(bx, by) == name


def test_schedule_parameters():
    layers = circuit.schedule()
    kinds = [l.kind for l in layers]
    assert kinds[:6] == ["neel", "spin_zz", "spin_x", "spin_yy", "spin_x", "inject"]
    pulses = [(l.k, l.j, l.dt) for l in layers if l.stage.startswith("pulse") and l.kind == "hopping"]
    assert pulses == [(-0.7511, -0.47413, 0.375)] * 2 + [(0.7511, -0.47413, 0.375)] * 2
    onsite = [l.angle for l in layers if l.kind == "onsite"]
    assert onsite == [-2 * 0.48897462, -2 * 0.48897462, 1.5, 3.0, 1.5]


def test_hopping_matrix_unitary():
    w = circuit.hopping_matrix(6, 6, -1, 1, 0.7511, -0.47413)
    assert np.allclose(w.conj().T @ w, np.eye(72), atol=1e-13)


def test_conventions():
    assert circuit.pairing_from_moments(3.0, 1.0, 8) == 0.5
    assert circuit.paper_convention(0.5) == -0.5
    assert circuit.issue_convention(0.5) == -0.25


def _circuit_file():
    for d in (os.environ.get("HUBBARD259_CIRCUITS"), ROOT / "data" / "circuits"):
        if d and (pathlib.Path(d) / "superconducting_pairing_2steps.py").exists():
            return pathlib.Path(d) / "superconducting_pairing_2steps.py"
    return None


@pytest.mark.skipif(_circuit_file() is None, reason="run scripts/fetch_circuits.sh first")
def test_validate_two_step_circuit():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_circuit.py"), str(_circuit_file())],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
