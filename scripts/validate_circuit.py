#!/usr/bin/env python3
"""Check the authors' two-step Guppy circuit against hubbard259.circuit.

The Guppy source is parsed statically (``ast``; nothing is imported or
executed).  Native gate conventions: ``Rz(a) = exp(-i pi a Z/2)``,
``ZZPhase(a) = exp(-i pi a ZZ/2)``,
``PhasedX(a, b) = exp[-i pi a (cos(pi b) X + sin(pi b) Y)/2]``, so a native
gate rotates by ``theta = pi a / 2``.  Gates whose angle is a multiple of
``pi/4`` (``a`` a multiple of 1/2) are Clifford: they implement the local
fermionic encoding, basis changes, the toric-code ancilla preparation and the
readout gadget.  Every remaining (non-Clifford) gate is reduced to its residual
angle ``theta mod pi/4`` and assigned to a layer of
:func:`hubbard259.circuit.schedule` by the magnitude of that residual:

=====================  ====================================  ===============
layer                  expected residual                     gates per unit
=====================  ====================================  ===============
spin ZZ / YY           0.12060681 / 0.05188793               1 per bond
staggered X            0.08211975, 0.33385306 (PhasedX)       1 per site
weak hopping           0.10667747 / 2                        2 per edge
weak onsite D(theta)   theta/4 = -0.97794924/4               1 per site
spin mixing            0.04714107 / 2                        2 per site
D(1.5), D(3)           0.375, 0.75 (mod pi/4)                1 per site
Gaussian pulse         0.375 |k + i j| / 2                   2 per edge
pulse phase gates      arg(k + i j) (PhasedX, mod pi/4)       -
=====================  ====================================  ===============

The check then requires (i) the layers to occur in the order of the schedule
(adjacent layers may interleave, as commuting gates on disjoint qubits are
reordered by the compiler), (ii) each layer to have exactly the expected
number of gates, and (iii) each layer to act on exactly the expected qubit
pairs: system qubit ``s`` (``s + 36``) carries the up (down) mode of site
``s = x + 6 y``; hopping layers must act on the edges of the corresponding
plaquette round(s) of :func:`hubbard259.lattice.plaquette_edges`, onsite and
spin-mixing layers on ``(s, s+36)``, spin layers on
:func:`hubbard259.lattice.spin_bonds`.  Boundary signs are not visible here:
the circuit realizes all four sectors through the ancilla register.
Rotation signs and the Clifford frames of individual gates are not compared
(they depend on the local encoding); only magnitudes, order, counts and
supports are.

Usage::

    scripts/fetch_circuits.sh data/circuits
    python scripts/validate_circuit.py data/circuits/superconducting_pairing_2steps.py

Exit status 0 means every check passed.
"""
import argparse
import ast
import math
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hubbard259 import circuit  # noqa: E402
from hubbard259.lattice import plaquette_edges, spin_bonds  # noqa: E402

LX = LY = 6
NS = LX * LY
QUARTER = math.pi / 4
TOL = 1e-8
GATES = {"zz_phase", "phased_x", "rz"}


def parse(path):
    """Ordered native gate list ``(name, qubits, half-turn parameters, line)``."""
    tree = ast.parse(pathlib.Path(path).read_text())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in GATES:
            qubits, params = [], []
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    qubits.append(int(arg.id.lstrip("q")))
                elif isinstance(arg, ast.Call) and getattr(arg.func, "id", "") == "angle":
                    params.append(float(ast.literal_eval(arg.args[0])))
                else:
                    raise ValueError(f"line {node.lineno}: unexpected argument {ast.dump(arg)}")
            calls.append((node.lineno, node.col_offset, node.func.id, tuple(qubits), tuple(params)))
    calls.sort()
    alloc = sum(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "qubit" for n in ast.walk(tree))
    return [(name, q, p, line) for line, _, name, q, p in calls], alloc


def residual(theta):
    """theta folded into [-pi/8, pi/8] modulo pi/4."""
    return math.remainder(theta, QUARTER)


def expected_layers():
    """Layer list: (label, gate kind, residual magnitude, expected qubit-pair Counter or None)."""
    edges = plaquette_edges(LX, LY)
    qubit = lambda m: m // 2 + NS * (m % 2)  # noqa: E731
    def hop_pairs(rounds):
        return Counter({frozenset((qubit(e.a), qubit(e.b))): 2 for e in edges if e.round in rounds})
    onsite_pairs = lambda mult: Counter({frozenset((s, s + NS)): mult for s in range(NS)})  # noqa: E731
    bonds = Counter(frozenset(b) for b in spin_bonds(LX, LY))
    sites = Counter({frozenset((s,)): 1 for s in range(NS)})
    out = []
    layers = circuit.schedule()
    i = 0
    while i < len(layers):
        layer = layers[i]
        if layer.kind == "spin_zz":
            out.append(("spin ZZ", "zz", layer.angle, bonds))
        elif layer.kind == "spin_yy":
            out.append(("spin YY", "zz", layer.angle, bonds))
        elif layer.kind == "spin_x":
            out.append((f"staggered X {layer.angle}", "px", layer.angle, sites))
        elif layer.kind == "hopping" and layer.stage == "weak":
            out.append((f"weak hopping round {layer.round}", "zz", layer.k * layer.dt / 2,
                        hop_pairs({layer.round})))
        elif layer.kind == "hopping":
            # both rounds of a pulse form one layer (same angles)
            rounds = {layer.round}
            while i + 1 < len(layers) and layers[i + 1].kind == "hopping" and layers[i + 1].stage == layer.stage:
                i += 1
                rounds.add(layers[i].round)
            out.append((f"{layer.stage} hopping", "zz", layer.dt * abs(complex(layer.k, layer.j)) / 2,
                        hop_pairs(rounds)))
            out.append((f"{layer.stage} phases", "px", math.atan2(abs(layer.j), abs(layer.k)), None))
        elif layer.kind == "onsite":
            out.append((f"{layer.stage} D({layer.angle:.8g})", "zz", layer.angle / 4, onsite_pairs(1)))
        elif layer.kind == "spin_mixing":
            out.append(("spin mixing", "zz", layer.angle / 2, onsite_pairs(2)))
        i += 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("guppy", type=pathlib.Path, help="superconducting_pairing_2steps.py")
    a = p.parse_args()
    gates, alloc = parse(a.guppy)
    census = Counter(g[0] for g in gates)
    print(f"{alloc} qubits allocated; native gates: " + ", ".join(f"{k} {v}" for k, v in sorted(census.items())))
    failures = []

    layers = expected_layers()
    mags = {}   # (kind, |residual|) -> layer indices; equal values share a key
    for idx, (_, kind, theta, _) in enumerate(layers):
        mag = abs(residual(theta))
        key = next((k for k in mags if k[0] == kind and abs(k[1] - mag) < TOL), (kind, mag))
        mags.setdefault(key, []).append(idx)

    # classify every non-Clifford gate
    stream = []
    for name, qubits, params, line in gates:
        theta = math.pi * params[0] / 2
        r = residual(theta)
        if abs(r) < TOL:
            continue
        kind = {"zz_phase": "zz", "phased_x": "px", "rz": "rz"}[name]
        if any(q >= 2 * NS for q in qubits):
            failures.append(f"line {line}: non-Clifford {name} on ancilla qubit {qubits}")
        key = next((k for k in mags if k[0] == kind and abs(k[1] - abs(r)) < TOL), None)
        if key is None:
            failures.append(f"line {line}: {name}{params} residual angle {r:.10f} matches no layer")
            continue
        stream.append((key, qubits, r, params, line))

    # sequential assignment (adjacent layers may interleave)
    need = [None if pairs is None else sum(pairs.values()) for _, _, _, pairs in layers]
    got = [Counter() for _ in layers]
    signs = [[] for _ in layers]
    cur = 0
    for key, qubits, r, params, line in stream:
        # advance past completed counted layers and past uncounted (phase) layers
        while cur < len(layers) and (need[cur] is None or sum(got[cur].values()) == need[cur]):
            cur += 1
        lo = cur - 1 if cur > 0 and need[cur - 1] is None else cur
        candidates = [i for i in mags[key] if lo <= i <= cur + 2 and
                      (need[i] is None or sum(got[i].values()) < need[i])]
        if not candidates:
            failures.append(f"line {line}: gate of class {key} out of schedule order (at layer {layers[cur][0]})")
            break
        i = candidates[0]
        got[i][frozenset(qubits)] += 1
        signs[i].append((qubits, r, params))

    print(f"{len(stream)} non-Clifford gates assigned to {len(layers)} schedule layers:")
    for i, (label, kind, theta, pairs) in enumerate(layers):
        n = sum(got[i].values())
        status = "ok"
        if pairs is not None and got[i] != pairs:
            status = "MISMATCH"
            failures.append(f"layer {label}: {n} gates, expected {sum(pairs.values())} on the expected qubit pairs")
        if pairs is None and n == 0:
            status = "MISSING"
            failures.append(f"layer {label}: no gates")
        print(f"  {label:28s} {kind}  |theta| mod pi/4 = {abs(residual(theta)):.10f}  gates {n:4d}  {status}")
        if label.endswith("phases"):
            phi = theta / math.pi
            bad = [pp for _, _, pp in signs[i] if min(abs(math.remainder(pp[1] - s * phi, 0.5)) for s in (1, -1)) > TOL]
            if bad:
                failures.append(f"layer {label}: phase parameter not +-arg(k+ij)/pi mod 1/2: {bad[:3]}")

    if failures:
        print("FAIL")
        for f in failures[:20]:
            print("  " + f)
        return 1
    print("PASS: parameters, layer order, gate counts and qubit supports match hubbard259.circuit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
