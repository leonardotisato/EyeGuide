#!/usr/bin/env python3
"""Resource-selection helpers for FINN memory placement experiments.

These helpers are intentionally dependency-free. The build step can use them
inside FINN, and local unit tests can validate the shape/ranking logic without
requiring a FINN Docker environment.
"""

import math
from typing import Dict, Iterable, List, Mapping, Sequence


BRAM36_DATA_MODES = (
    (9, 4096),
    (18, 2048),
    (36, 1024),
    (72, 512),
)


def estimate_bram18_sites(depth: int, width: int, banks: int = 1) -> int:
    """Estimate RAMB18-equivalent sites using BRAM36 data-width modes."""

    if depth <= 0 or width <= 0 or banks <= 0:
        return 0
    bram36_sites = min(
        banks * math.ceil(depth / float(mode_depth)) * math.ceil(width / float(mode_width))
        for mode_width, mode_depth in BRAM36_DATA_MODES
    )
    return int(bram36_sites * 2)


def estimate_lutram64_sites(depth: int, width: int, banks: int = 1) -> int:
    """Estimate LUTRAM64x1 primitives for a banked logical memory."""

    if depth <= 0 or width <= 0 or banks <= 0:
        return 0
    return int(banks * math.ceil(depth / 64.0) * width)


def estimate_fifo_bram18_sites(depth: int, width: int) -> int:
    """Estimate RAMB18-equivalent sites for an XPM FIFO memory shape."""

    if depth <= 0 or width <= 0:
        return 0
    if width == 1:
        return int(math.ceil(depth / 16384.0))
    if width == 2:
        return int(math.ceil(depth / 8192.0))
    if width <= 4:
        return int(math.ceil(depth / 4096.0) * math.ceil(width / 4.0))
    if width <= 9:
        return int(math.ceil(depth / 2048.0) * math.ceil(width / 9.0))
    if width <= 18 or depth > 512:
        return int(math.ceil(depth / 1024.0) * math.ceil(width / 18.0))
    return int(math.ceil(depth / 512.0) * math.ceil(width / 36.0))


def estimate_fifo_lutram_luts(depth: int, width: int) -> int:
    """Estimate LUTRAM pressure for a distributed XPM FIFO."""

    if depth <= 0 or width <= 0:
        return 0
    addr_luts = 2 * math.ceil(math.log(depth, 2))
    memory_luts = math.ceil(depth / 32.0) * math.ceil(width / 2.0)
    return int(addr_luts + memory_luts)


def estimate_fifo_uram_sites(depth: int, width: int) -> int:
    """Exact URAM288 site count for a logical FIFO memory shape."""

    if depth <= 0 or width <= 0:
        return 0
    return int(math.ceil(depth / 4096.0) * math.ceil(width / 72.0))


def _candidate_with_costs(candidate: Mapping) -> Dict:
    depth = int(candidate.get("depth", 0) or 0)
    width = int(candidate.get("width", 0) or 0)
    enriched = dict(candidate)
    enriched["depth"] = depth
    enriched["width"] = width
    enriched["bram18"] = int(
        candidate.get("bram18", estimate_fifo_bram18_sites(depth, width)) or 0
    )
    enriched["uram"] = int(
        candidate.get("uram", estimate_fifo_uram_sites(depth, width)) or 0
    )
    enriched["slicem"] = int(
        candidate.get(
            "slicem",
            candidate.get("lutram", estimate_fifo_lutram_luts(depth, width)),
        )
        or 0
    )
    return enriched


def _choose_knapsack(
    candidates: Sequence[Mapping],
    budget: int,
    value_key: str,
    cost_key: str,
) -> List[Dict]:
    if budget <= 0:
        return []

    dp = {0: (0, [])}
    for candidate in candidates:
        cost = int(candidate.get(cost_key, 0) or 0)
        value = int(candidate.get(value_key, 0) or 0)
        if cost <= 0 or value <= 0 or cost > budget:
            continue
        next_dp = dict(dp)
        for used, (prev_value, prev_items) in dp.items():
            new_used = used + cost
            if new_used > budget:
                continue
            new_value = prev_value + value
            if new_used not in next_dp or new_value > next_dp[new_used][0]:
                next_dp[new_used] = (new_value, prev_items + [dict(candidate)])
        dp = next_dp

    best_key = max(dp, key=lambda key: (dp[key][0], -key))
    return dp[best_key][1]


def choose_fifo_uram_relievers(
    candidates: Iterable[Mapping],
    total_uram_budget: int,
    forced_min_depth: int = 8192,
    min_slicem_per_uram: float = 128.0,
) -> List[Dict]:
    """Choose FIFO candidates for URAM.

    Deep FIFOs are forced first because they relieve the BRAM bottleneck. The
    leftover URAM budget is then spent on shapes that relieve the most
    LUTRAM/SRL pressure per URAM site, which avoids wide/shallow traps such as
    32x1152 FIFOs.
    """

    enriched = [
        {**_candidate_with_costs(candidate), "_order": idx}
        for idx, candidate in enumerate(candidates)
    ]
    eligible = [item for item in enriched if item["uram"] > 0]

    forced_pool = [
        item
        for item in eligible
        if item["depth"] >= forced_min_depth and item["bram18"] > 0
    ]
    forced = _choose_knapsack(forced_pool, total_uram_budget, "bram18", "uram")
    forced_ids = {item["_order"] for item in forced}
    used_uram = sum(item["uram"] for item in forced)
    remaining_budget = max(total_uram_budget - used_uram, 0)

    opportunistic_pool = []
    for item in eligible:
        if item["_order"] in forced_ids:
            continue
        score = item["slicem"] / float(item["uram"])
        if score < min_slicem_per_uram:
            continue
        opportunistic_pool.append({**item, "slicem_per_uram": score})

    opportunistic_pool.sort(
        key=lambda item: (
            -item["slicem_per_uram"],
            -item["slicem"],
            item["uram"],
            item["_order"],
        )
    )
    opportunistic = _choose_knapsack(
        opportunistic_pool, remaining_budget, "slicem", "uram"
    )

    selected = forced + opportunistic
    selected.sort(key=lambda item: item["_order"])
    for item in selected:
        item.pop("_order", None)
    return selected


def choose_bram_to_lutram_relievers(
    candidates: Iterable[Mapping],
    target_bram18: int,
    max_lutram: int,
) -> List[Dict]:
    """Choose BRAM-backed memories to move to LUTRAM.

    Prefer the lowest-LUTRAM subset that reaches the target BRAM relief. If no
    subset reaches the target, return the highest-relief subset within budget.
    """

    enriched = []
    for idx, candidate in enumerate(candidates):
        bram18 = int(candidate.get("bram18", 0) or 0)
        lutram = int(candidate.get("lutram", 0) or 0)
        if bram18 <= 0 or lutram <= 0 or lutram > max_lutram:
            continue
        enriched.append({**dict(candidate), "bram18": bram18, "lutram": lutram, "_order": idx})

    dp = {0: (0, [])}
    for candidate in enriched:
        cost = candidate["lutram"]
        value = candidate["bram18"]
        next_dp = dict(dp)
        for used, (prev_value, prev_items) in dp.items():
            new_used = used + cost
            if new_used > max_lutram:
                continue
            new_value = prev_value + value
            if new_used not in next_dp or new_value > next_dp[new_used][0]:
                next_dp[new_used] = (new_value, prev_items + [candidate])
        dp = next_dp

    reaching = [
        (used, value, items)
        for used, (value, items) in dp.items()
        if value >= target_bram18
    ]
    if reaching:
        _, _, selected = min(
            reaching,
            key=lambda item: (
                item[0],
                len(item[2]),
                -item[1],
                [cand["_order"] for cand in item[2]],
            ),
        )
    else:
        _, (best_value, selected) = max(
            dp.items(),
            key=lambda item: (
                item[1][0],
                -item[0],
                -len(item[1][1]),
            ),
        )
        if best_value <= 0:
            selected = []

    selected = [dict(item) for item in selected]
    selected.sort(key=lambda item: item["_order"])
    for item in selected:
        item.pop("_order", None)
    return selected


def choose_lutram_budget_relievers(
    candidates: Iterable[Mapping],
    max_lutram: int,
) -> List[Dict]:
    """Choose all useful LUTRAM candidates that fit in the LUTRAM budget.

    This is for cases where the BRAM estimate is only a priority score, not a
    reliable stopping condition. It maximizes the score under budget instead of
    stopping as soon as a target is reached.
    """

    enriched = []
    for idx, candidate in enumerate(candidates):
        bram18 = int(candidate.get("bram18", 0) or 0)
        lutram = int(candidate.get("lutram", 0) or 0)
        if bram18 <= 0 or lutram <= 0 or lutram > max_lutram:
            continue
        enriched.append({**dict(candidate), "bram18": bram18, "lutram": lutram, "_order": idx})

    selected = _choose_knapsack(enriched, max_lutram, "bram18", "lutram")
    selected = [dict(item) for item in selected]
    selected.sort(key=lambda item: item["_order"])
    for item in selected:
        item.pop("_order", None)
    return selected
