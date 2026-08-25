"""Sub-layer fragment execution — split one layer's weight matrix across
heterogeneous-capacity workers, none of which needs to hold the whole thing.

WHAT IS PROVEN HERE, TODAY, ON THIS MACHINE: a linear layer (the shape of both halves
of a transformer MLP block) can be split column-wise or row-wise across N workers,
each given a fragment sized proportional to its measured capacity, and the combined
result matches a full unsplit reference to a tight numerical tolerance. `tests/
test_fragment.py` runs this at random sizes with random capacities and asserts and
PRINTS the measured relative error on every run — never a claimed number, always a
freshly computed one.

WHAT IS NOT CLAIMED: this is CPU-only numpy, not a GPU kernel, not wired to any real
model checkpoint (Kimi K3 or otherwise), and the capacity-based allocation is not
tested against real rented hardware. It is the mechanism, proven small, honestly
labelled. A prior K3-scale physical run (four real 8-12GB consumer GPUs, ~1e-8
relative error on a real ~19.1GB layer) was reported in conversation but its source
and artifacts could not be located on this machine or recovered from the reporter's
own memory as of 2026-08-24 -- so this module does not claim to reproduce that run.
It is a fresh, independently-verified implementation of the same idea.

THE TWO SPLITS (standard tensor-parallel terminology, same shape Megatron-LM uses):

  column-parallel: split the weight matrix along its OUTPUT dimension. Each worker
    holds a subset of output rows, computes its slice of the output independently,
    and the combine step is a CONCATENATION (no cross-worker communication needed
    mid-computation). Use for the up-projection half of an MLP block.

  row-parallel: split the weight matrix along its INPUT dimension. Each worker holds
    a subset of input columns, needs the corresponding slice of the input activation,
    computes a PARTIAL output, and the combine step is a SUM across workers. Use for
    the down-projection half of an MLP block, chained after a column-parallel
    up-projection, so the two splits compose without an all-gather between them.

CAPACITY-PROPORTIONAL ALLOCATION: `plan_fragments` is the largest-remainder proportional
distribution shard's own `Scheduler._distribute` uses (E:/repos/intake/shard-master/
shard-master/shard/scheduler.py, Apache-2.0) -- hand out `total` units across workers
proportional to their capacity, never exceeding a worker's cap, closing exactly on
`total`. Reimplemented here rather than imported: shard is a separate, unvendored
project (see the dossier's ADOPT-not-vendor guidance for anything over a few hundred
lines), and this file is small enough that a clean reimplementation with its own tests
is more honest than a partial import of someone else's package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ============================================================================
# Capacity-proportional fragment planning
# ============================================================================

@dataclass(frozen=True)
class WorkerCapacity:
    """One worker's declared capacity, in whatever unit the caller measures (VRAM
    bytes, max output rows it can hold, etc.) -- `plan_fragments` is unit-agnostic."""

    worker_id: str
    capacity: int

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"{self.worker_id}: capacity must be >= 0, got {self.capacity}")


def plan_fragments(total_units: int, workers: list[WorkerCapacity]) -> dict[str, int]:
    """Distribute `total_units` across `workers` proportional to capacity, never
    exceeding a worker's own capacity, summing to exactly `total_units`.

    Largest-remainder rounding: each worker gets floor(total * cap / capsum) first,
    then the leftover is handed to whichever worker has the largest fractional
    remainder and still has spare capacity, repeating until nothing is left over.
    Raises ValueError if sum(capacity) < total_units -- the pool genuinely cannot
    hold the layer, and that must fail loudly, not silently under-allocate.
    """
    if total_units <= 0:
        raise ValueError(f"total_units must be > 0, got {total_units}")
    if not workers:
        raise ValueError("no workers given")

    capsum = sum(w.capacity for w in workers)
    if capsum < total_units:
        raise ValueError(
            f"pool capacity {capsum} < required {total_units} -- this layer does not fit"
        )

    ids = [w.worker_id for w in workers]
    cap = {w.worker_id: w.capacity for w in workers}

    base = {i: min(cap[i], (total_units * cap[i]) // capsum) for i in ids}
    assigned = sum(base.values())
    remainder = total_units - assigned

    # order by largest fractional remainder first, tie-broken by largest capacity
    # (a bigger worker absorbs rounding slop more comfortably than a small one)
    order = sorted(ids, key=lambda i: (-(total_units * cap[i] % capsum), -cap[i]))

    k = 0
    safety_limit = 4 * len(order) * (total_units + 1)  # never trips when capsum >= total
    while remainder > 0:
        i = order[k % len(order)]
        if base[i] < cap[i]:
            base[i] += 1
            remainder -= 1
        k += 1
        if k > safety_limit:
            raise AssertionError(
                "plan_fragments: remainder did not close -- this is a bug, not a capacity "
                "shortfall (capsum >= total_units was already checked above)"
            )

    return base


# ============================================================================
# Column-parallel: split along the OUTPUT dimension, combine by concatenation
# ============================================================================

@dataclass(frozen=True)
class ColumnFragment:
    """One worker's slice of a column-parallel weight matrix: rows [row_start, row_end)
    of the full (out_features, in_features) weight."""

    worker_id: str
    row_start: int
    row_end: int
    weight_slice: np.ndarray  # shape (row_end - row_start, in_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, in_features) -> partial output (batch, row_end - row_start)."""
        return x @ self.weight_slice.T


def split_column_parallel(
    weight: np.ndarray, workers: list[WorkerCapacity]
) -> list[ColumnFragment]:
    """Split `weight` (out_features, in_features) into per-worker row slices,
    capacity-unit == output rows. Fragments are returned in worker order; combining
    them is a concatenation along axis=1 of each fragment's forward() output, in the
    SAME order plan_fragments assigned row ranges -- see execute_column_parallel."""

    out_features = weight.shape[0]
    allocation = plan_fragments(out_features, workers)

    fragments: list[ColumnFragment] = []
    row = 0
    for w in workers:  # preserve caller's worker order, not allocation dict order
        n_rows = allocation[w.worker_id]
        if n_rows == 0:
            continue
        fragments.append(
            ColumnFragment(
                worker_id=w.worker_id,
                row_start=row,
                row_end=row + n_rows,
                weight_slice=weight[row : row + n_rows, :],
            )
        )
        row += n_rows
    assert row == out_features, f"fragment rows summed to {row}, expected {out_features}"
    return fragments


def execute_column_parallel(x: np.ndarray, fragments: list[ColumnFragment]) -> np.ndarray:
    """Run every fragment's forward() and concatenate in row-range order -> full
    (batch, out_features) output, with NO cross-worker communication mid-computation."""
    ordered = sorted(fragments, key=lambda f: f.row_start)
    parts = [f.forward(x) for f in ordered]
    return np.concatenate(parts, axis=1)


# ============================================================================
# Row-parallel: split along the INPUT dimension, combine by summation
# ============================================================================

@dataclass(frozen=True)
class RowFragment:
    """One worker's slice of a row-parallel weight matrix: columns [col_start, col_end)
    of the full (out_features, in_features) weight, and the matching input slice."""

    worker_id: str
    col_start: int
    col_end: int
    weight_slice: np.ndarray  # shape (out_features, col_end - col_start)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, col_end - col_start) -- ALREADY sliced to this worker's input
        columns -> partial output (batch, out_features), summed with every other
        worker's partial by the caller."""
        return x @ self.weight_slice.T


def split_row_parallel(weight: np.ndarray, workers: list[WorkerCapacity]) -> list[RowFragment]:
    """Split `weight` (out_features, in_features) into per-worker column slices,
    capacity-unit == input columns."""

    in_features = weight.shape[1]
    allocation = plan_fragments(in_features, workers)

    fragments: list[RowFragment] = []
    col = 0
    for w in workers:
        n_cols = allocation[w.worker_id]
        if n_cols == 0:
            continue
        fragments.append(
            RowFragment(
                worker_id=w.worker_id,
                col_start=col,
                col_end=col + n_cols,
                weight_slice=weight[:, col : col + n_cols],
            )
        )
        col += n_cols
    assert col == in_features, f"fragment cols summed to {col}, expected {in_features}"
    return fragments


def execute_row_parallel(x: np.ndarray, fragments: list[RowFragment]) -> np.ndarray:
    """Slice x to each fragment's input columns, run forward(), and SUM the partials
    -> full (batch, out_features) output. Summation, not concatenation, is what makes
    row-parallel compose with a preceding column-parallel stage without an all-gather
    in between (each worker's partial is already a full-width, partially-summed
    contribution to the final output)."""
    ordered = sorted(fragments, key=lambda f: f.col_start)
    total = None
    for f in ordered:
        x_slice = x[:, f.col_start : f.col_end]
        partial = f.forward(x_slice)
        total = partial if total is None else total + partial
    assert total is not None
    return total


# ============================================================================
# Reference verification -- every fragmented run is checked against the unsplit
# computation, never trusted on its own. Same discipline as every prior-art repo
# this package's design was informed by.
# ============================================================================

def relative_error(fragmented: np.ndarray, reference: np.ndarray) -> float:
    """||fragmented - reference|| / ||reference||, the same metric quoted throughout
    this session's research (E025's reported ~1e-8, shard's cosine/rel-error spot
    checks). Returns +inf if the reference norm is exactly zero (division-by-zero
    guard, not a silent 0.0 that would misreport a degenerate all-zero case as a
    perfect match)."""
    ref_norm = np.linalg.norm(reference)
    if ref_norm == 0.0:
        return float("inf") if np.linalg.norm(fragmented) != 0.0 else 0.0
    return float(np.linalg.norm(fragmented - reference) / ref_norm)
