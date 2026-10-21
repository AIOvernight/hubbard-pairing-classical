"""CPU tests of the connected-cumulant hierarchy (no GPU required)."""
import math

import numpy as np
import pytest

from hubbard259._geometry import first_propagator
from hubbard259.clusters import (connected_supports, incidence_weights, pp_gauge_map, prepare_cluster,
                                 translation_map)
from hubbard259.cumulants import (comb_modes, entry_moments, forward_cumulants, inverse_moments, raw_factors,
                                  third_cumulant)
from hubbard259.production import run_hierarchy
from hubbard259.projection import compound_rect, project_one
from hubbard259.readout import region_orbits
from hubbard259.reduction import fock_reduction_check


@pytest.mark.parametrize('bx,by', [(1, 1), (-1, -1)])
def test_dressed_operator_matches_direct_fock_4x2(bx, by):
    direct, dressed, monomials = fock_reduction_check(4, 2, bx, by)
    assert monomials == 35456
    assert abs(direct - dressed) < 1e-12


def test_census_6x6():
    supports = connected_supports(6, 6, 6)
    counts = [sum(len(s) == k for s in supports) for k in range(1, 7)]
    assert counts[:5] == [36, 72, 216, 684, 2268]
    assert len(supports) == 10992 and counts[5] == 7716
    orbits = [len(region_orbits(6, 6, s)[0]) for s in range(1, 5)]
    assert orbits == [4, 70, 804, 6545] and sum(orbits) == 7423
    assert sum(sum(region_orbits(6, 6, s)[1]) for s in range(1, 5)) == 66711


def test_incidence_weights_partition_anchor():
    supports = connected_supports(4, 4, 5)
    w = incidence_weights(supports)
    for p in range(16):
        assert sum(v for s, v in w.items() if p in s) == 1
    # Equivalent to explicit connected-subcluster subtraction.
    rng = np.random.default_rng(1)
    raw = {s: rng.normal() for s in w}
    delta = {}
    for s in sorted(w, key=len):
        delta[s] = raw[s] - sum(delta[t] for k in range(1, len(s)) for t in _subsets(s, k) if t in delta)
    assert math.isclose(sum(delta.values()), sum(w[s] * raw[s] for s in w), rel_tol=0, abs_tol=1e-9)


def _subsets(s, k):
    import itertools
    return itertools.combinations(s, k)


def test_cluster_state_normalised():
    state = prepare_cluster(6, 6, (0, 1, 6, 7))
    assert abs(np.vdot(state, state) - 1) < 1e-13


def test_moment_cumulant_roundtrip():
    rec = raw_factors(4, 4, (0, 1, 4), 1, 1)
    tables = [np.ones((1, 1), complex)] + [entry_moments(e) for e in rec]
    kap = inverse_moments(tables, 6, 6)
    back, _ = forward_cumulants(kap, 6, 6)
    for r in range(1, 7):
        assert np.max(abs(back[r - 1] - tables[r])) < 1e-13


def _record(lx, ly, sites, bx, by):
    return dict(sites=tuple(sites), modes=np.array([2 * s + t for s in sites for t in (0, 1)]),
                phases=np.ones(2 * len(sites)), entries=raw_factors(lx, ly, sites, bx, by))


@pytest.mark.parametrize('bx,by', [(-1, 1), (-1, -1)])
def test_shared_sources_match_direct_preparation(bx, by):
    lx = ly = 4
    w = first_propagator(lx, ly, bx, by)
    output = [2 * p + t for p in (3, 4, 7) for t in (0, 1)]
    # Gauge-shared PP source (no winding) and translated twisted source (winding row).
    for source, dx, dy, mapper, src_boundary in [((0, 3, 4), 0, 0, pp_gauge_map, (1, 1)),
                                                  ((0, 3, 4), 2, 2, pp_gauge_map, (1, 1)),
                                                  ((0, 1, 2, 3), 2, 0, translation_map, (bx, by))]:
        mapping = mapper(source, lx, ly, dx, dy, bx, by)
        shared = dict(sites=mapping['sites'], modes=np.asarray(mapping['modes']),
                      phases=np.asarray(mapping['phases'], float),
                      entries=raw_factors(lx, ly, source, *src_boundary))
        direct = _record(lx, ly, mapping['sites'], bx, by)
        for a, b in zip(project_one(shared, w, output), project_one(direct, w, output)):
            assert np.max(abs(a - b)) < 1e-13


def test_kappa3_projection_is_covariant():
    """Projecting local kappa_3 equals the third cumulant of projected raw moments."""
    lx = ly = 4
    rec = _record(lx, ly, (1, 2, 5, 6), 1, 1)
    w = first_propagator(lx, ly, 1, 1)
    output = [2 * p + t for p in (1, 2, 6, 10) for t in (0, 1)]
    ww = w[np.ix_(output, rec['modes'])]
    u3 = compound_rect(np.ascontiguousarray(ww), comb_modes(8, 3), comb_modes(8, 3))
    projected_k3 = u3.conj() @ third_cumulant(rec) @ u3.T
    reference = project_one(rec, w, output, 3)[2]
    assert np.max(abs(projected_k3 - reference)) < 1e-14


def test_hierarchy_4x2_pp_regions_match_reference():
    # Complex128 reference rows of the stored 4x2 PP control (rank 6, cutoff 4).
    reference = {(0, 1): 0.020034413322950972, (0, 1, 5): 0.0019234678343577716,
                 (0, 1, 4, 5): 0.0027465882326132506, (0, 2, 5, 7): 3.6613056466778644e-05}
    result = run_hierarchy(4, 2, 1, 1, ranks=(6,), cutoffs=(4,), device='cpu', regions=list(reference))
    assert result['status'] == 'partial' and result['regions'] == 4
    for row in result['rows']:
        value = complex(*row['values']['r6c4'])
        assert abs(value - reference[tuple(row['region'])]) < 1e-14
