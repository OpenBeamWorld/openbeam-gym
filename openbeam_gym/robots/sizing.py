# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Fixed-step sizing shared by the robot builders.

The game integrates every node explicitly at a fixed 0.5 ms step ``H`` and
holds each node to the criterion ``omega^2 h^2 + 2 c h / m <= 4``. At spawn it
checks two shares of that criterion per node:

* beams: ``sum (2 h^2 k + 4 h c) / m_i`` over the beams on node i; the game
  lowers the beams on a node above ``BEAM_GAME_LIMIT`` (2.6);
* torsion bars: ``sum (h^2 k + 2 h c) |g_i| sum_j |g_j| / m_i`` over the bars
  on node i, with g the gradient of the bar's dihedral angle with respect to
  its four nodes; the game scales the bars on a node above
  ``TORSION_GAME_LIMIT`` (0.8) down to it.

Explicit damping is the expensive part (``2 h c`` against ``h^2 k``: at
h = 0.5 ms a unit of damping costs 4000 units of stiffness), and a bar's
budget grows with its lever nodes' mass times lever squared.

The builders size every beam and bar below these caps themselves, so the game
scales nothing at spawn: ``size_beams`` gives each beam an equal slice of its
nodes' beam ceiling, ``size_torsion`` shares the torsion ceiling out by
progressive filling (in proportion to each joint's need) and then splits each
joint's share between spring and damper for a chosen damping ratio against the
inertia the joint moves. ``node_loads`` evaluates both shares for any robot,
including ones not sized here (the test arm).

Stdlib only.
"""

from __future__ import annotations

import math

H = 0.0005                 # s, the game's fixed physics step
GRAVITY = 9.81
BEAM_GAME_LIMIT = 2.6      # per-node beam share above which the game lowers beams
TORSION_GAME_LIMIT = 0.8   # per-node torsion share above which the game scales bars


# -- vectors -------------------------------------------------------------------------

def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def add(*vs):
    return tuple(sum(c) for c in zip(*vs))


def mul(v, s):
    return tuple(c * s for c in v)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    return tuple(c / n for c in a)


def perpendicular(p, center, axis):
    """The part of p - center perpendicular to the unit axis."""
    d = sub(p, center)
    return sub(d, mul(axis, dot(d, axis)))


def sig_floor(x: float, digits: int = 3) -> float:
    """x rounded down to `digits` significant digits (stays inside a budget)."""
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x)) - digits + 1
    return round(math.floor(x / 10 ** e + 1e-9) * 10 ** e, max(0, -e))


# -- torsion bar geometry (the engine's angle gradient) --------------------------------

def dihedral_gradient(p1, p2, p3, p4):
    """Gradient of the dihedral angle of lever p1 and lever p4 about p2 -> p3
    with respect to each node (Blondel and Karplus); they sum to zero."""
    f, g, h = sub(p1, p2), sub(p2, p3), sub(p4, p3)
    a, b = cross(f, g), cross(h, g)
    aa, bb, gl = dot(a, a), dot(b, b), norm(g)
    fg, hg = dot(f, g), dot(h, g)
    g1 = mul(a, -gl / aa)
    g4 = mul(b, gl / bb)
    g2 = add(mul(a, gl / aa), mul(a, fg / (aa * gl)), mul(b, -hg / (bb * gl)))
    g3 = add(mul(b, hg / (bb * gl)), mul(a, -fg / (aa * gl)), mul(b, -gl / bb))
    return [g1, g2, g3, g4]


def bar_factors(positions: dict, masses: dict, ns) -> dict[str, float]:
    """|g_i| sum_j |g_j| / m_i per node of one bar (nodes ``ns``)."""
    mags = [norm(v) for v in dihedral_gradient(*(positions[n] for n in ns))]
    total = sum(mags)
    out: dict[str, float] = {}
    for n, m in zip(ns, mags):
        out[n] = out.get(n, 0.0) + m * total / masses[n]
    return out


# -- per-node loads ---------------------------------------------------------------------

def node_loads(positions: dict, masses: dict, beams: dict, hydros, h: float = H) -> dict[str, dict[str, float]]:
    """Per node: the beams' and the torsion bars' share of the fixed-step
    criterion, as asked (before any game scaling). ``beams`` maps (a, b) to
    (spring, damp); ``hydros`` is an iterable of (nodes, spring, damp)."""
    beam = {n: 0.0 for n in positions}
    for (a, b), (k, c) in beams.items():
        for n in (a, b):
            beam[n] += (2 * h * h * k + 4 * h * c) / masses[n]
    torsion = {n: 0.0 for n in positions}
    for ns, k, c in hydros:
        for n, f in bar_factors(positions, masses, ns).items():
            torsion[n] += (h * h * k + 2 * h * c) * f
    return {n: {"beams": beam[n], "torsion": torsion[n]} for n in positions}


# -- beams --------------------------------------------------------------------------------

def size_beams(pairs, masses: dict, ceiling: float, spring_share: float, spring_max: float,
               h: float = H) -> dict[tuple[str, str], tuple[float, float]]:
    """Every beam gets an equal slice of the beam ceiling of the busier of its
    two nodes (``ceiling * m / degree``), split ``spring_share`` to the spring
    (capped at ``spring_max``) and the rest to the damper."""
    degree: dict[str, int] = {}
    for a, b in pairs:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    out = {}
    for a, b in sorted(pairs):
        share = min(ceiling * masses[n] / degree[n] for n in (a, b))   # 2 h^2 k + 4 h c
        k = min(spring_max, spring_share * share / (2 * h * h))
        c = (share - 2 * h * h * k) / (4 * h)
        out[(a, b)] = (sig_floor(k), sig_floor(c))
    return out


# -- torsion bars --------------------------------------------------------------------------

def progressive_fill(needs: dict[int, float], factors: dict[int, dict[str, float]],
                     ceiling: float) -> dict[int, float]:
    """Progressive filling: every bar's coefficient grows in proportion to its
    need until a node it loads reaches `ceiling`; then it stops and the others
    go on. The bars on every node stay within the ceiling (max-min fair in
    need-normalised terms)."""
    active = set(needs)
    q = {b: 0.0 for b in needs}
    frozen_load: dict[str, float] = {}
    t = 0.0
    while active:
        rates: dict[str, float] = {}
        for b in active:
            for n, f in factors[b].items():
                rates[n] = rates.get(n, 0.0) + f * needs[b]
        best = None
        for n, rate in sorted(rates.items()):
            if rate <= 0:
                continue
            tn = (ceiling - frozen_load.get(n, 0.0)) / rate
            if best is None or tn < best[0] - 1e-15:
                best = (tn, n)
        t = best[0]
        saturated = {b for b in active if best[1] in factors[b]}
        # Freeze every bar on the saturated node (and any other node saturated at t).
        for n, rate in rates.items():
            if abs((ceiling - frozen_load.get(n, 0.0)) / rate - t) <= 1e-12 * max(1.0, t):
                saturated |= {b for b in active if n in factors[b]}
        for b in sorted(saturated):
            q[b] = needs[b] * t
            for n, f in factors[b].items():
                frozen_load[n] = frozen_load.get(n, 0.0) + f * q[b]
        active -= saturated
    return q


def inertia_about(positions: dict, masses: dict, nodes, center, axis) -> float:
    """sum m r^2 of `nodes` about the line through `center` along the unit `axis`."""
    total = 0.0
    for n in nodes:
        r = perpendicular(positions[n], center, axis)
        total += masses[n] * dot(r, r)
    return total


def gravity_torque_bound(moved_mass: float, inertia: float) -> float:
    """Worst-case gravity torque on a joint: the moved mass held out level at
    its radius of gyration about the joint axis."""
    return moved_mass * GRAVITY * math.sqrt(inertia / moved_mass)


def joint_need(inertia: float, torque: float, sag_target: float, zeta: float, k_floor: float,
               h: float = H) -> float:
    """The torsion coefficient (h^2 K + 2 h C) that holds `torque` within
    `sag_target` at damping ratio `zeta` against `inertia`."""
    k_need = max(k_floor, torque / sag_target)
    return h * h * k_need + 4 * h * zeta * math.sqrt(k_need * inertia)


def size_torsion(positions: dict, masses: dict, joints: list[dict], ceiling: float, zeta: float,
                 h: float = H) -> list[dict]:
    """Shares the per-node torsion ceiling out over the joints' bars and splits
    each joint's share into spring and damper for damping ratio `zeta`.

    Each joint is a dict with ``bars`` (lists of four nodes: parent lever,
    axis start, axis end, child lever), ``inertia`` (kg m^2 it moves) and
    ``need`` (``joint_need``). Returns, per joint in order, ``{"bars": [(k, c),
    ...], "spring": sum k, "damp": sum c}``; k and c are rounded down to three
    significant digits, so the asked loads stay within the ceiling."""
    bars, needs, factors = [], {}, {}
    for j in joints:
        for ns in j["bars"]:
            b = len(bars)
            bars.append((j, ns))
            needs[b] = j["need"] / len(j["bars"])
            factors[b] = bar_factors(positions, masses, ns)
    q = progressive_fill(needs, factors, ceiling)
    out = []
    for j in joints:
        ids = [b for b, (jj, _) in enumerate(bars) if jj is j]
        q_total = sum(q[b] for b in ids)
        inertia = j["inertia"]
        # h^2 K + 2 h C = Q with C = 2 zeta sqrt(K I): solve for sqrt(K).
        bq = 4 * h * zeta * math.sqrt(inertia)
        u = (-bq + math.sqrt(bq * bq + 4 * h * h * q_total)) / (2 * h * h)
        alpha = h * h * u * u / q_total
        rows, spring_total, damp_total = [], 0.0, 0.0
        for b in ids:
            k = sig_floor(alpha * q[b] / (h * h))
            c = sig_floor((1 - alpha) * q[b] / (2 * h))
            spring_total += k
            damp_total += c
            rows.append((k, c))
        out.append({"bars": rows, "spring": spring_total, "damp": damp_total})
    return out


def dynamics(spring: float, damp: float, inertia: float, torque_max: float) -> dict[str, float]:
    """Natural frequency, damping ratio and worst-case static sag of a joint."""
    return {"omega": math.sqrt(spring / inertia), "zeta": damp / (2 * math.sqrt(spring * inertia)),
            "gravity_sag_max": torque_max / spring}


# -- as the game reads it ------------------------------------------------------------------

def jbeam_physics(part: dict) -> tuple[dict, dict, dict, list]:
    """Positions, masses, beams {(a, b): (k, c)} and hydros [(nodes, k, c)]
    of one JBeam part, read the way the importer reads them: ``nodeWeight``,
    ``beamSpring`` and ``beamDamp`` modifier rows apply to the rows after them
    (and a row's own trailing options override them)."""
    positions, masses = {}, {}
    weight = None
    for row in part["nodes"][1:]:
        if isinstance(row, dict):
            weight = row.get("nodeWeight", weight)
            continue
        opts = row[4] if len(row) > 4 and isinstance(row[4], dict) else {}
        positions[row[0]] = tuple(float(v) for v in row[1:4])
        masses[row[0]] = float(opts.get("nodeWeight", weight))
    beams = {}
    spring = damp = None
    for row in part["beams"][1:]:
        if isinstance(row, dict):
            spring, damp = row.get("beamSpring", spring), row.get("beamDamp", damp)
            continue
        opts = row[2] if len(row) > 2 and isinstance(row[2], dict) else {}
        beams[(row[0], row[1])] = (float(opts.get("beamSpring", spring)), float(opts.get("beamDamp", damp)))
    hydros = [(row[:4], float(row[4]["spring"]), float(row[4]["damp"]))
              for row in part.get("torsionHydros", [])[1:] if isinstance(row, list)]
    return positions, masses, beams, hydros


def budget_report(part: dict) -> dict:
    """The worst per-node beam and torsion shares of one JBeam part, against
    the game's limits: what the game would lower or scale at spawn."""
    positions, masses, beams, hydros = jbeam_physics(part)
    loads = node_loads(positions, masses, beams, hydros)
    wb = max(loads, key=lambda n: (loads[n]["beams"], n))
    wt = max(loads, key=lambda n: (loads[n]["torsion"], n))
    over_t = sorted(n for n, v in loads.items() if v["torsion"] > TORSION_GAME_LIMIT)
    over_b = sorted(n for n, v in loads.items() if v["beams"] > BEAM_GAME_LIMIT)
    scales = []
    for ns, _k, _c in hydros:
        worst = max(loads[n]["torsion"] for n in ns)
        scales.append(min(1.0, TORSION_GAME_LIMIT / worst) if worst > 0 else 1.0)
    return {
        "worst_torsion": [wt, loads[wt]["torsion"]], "worst_beams": [wb, loads[wb]["beams"]],
        "torsion_nodes_over_limit": len(over_t), "beam_nodes_over_limit": len(over_b),
        "torsion_bars_scaled": sum(1 for s in scales if s < 1.0),
        "smallest_torsion_scale": min(scales) if scales else 1.0,
        "within_budget": not over_t and not over_b,
        "loads": loads,
    }
