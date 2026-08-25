"""Real, running proof of the sub-layer fragment mechanism -- not a claimed number,
a measured one, printed on every run so it can never silently drift stale."""

from __future__ import annotations

import numpy as np
import pytest
from awswarm.fragment import (
    WorkerCapacity,
    execute_column_parallel,
    execute_row_parallel,
    plan_fragments,
    relative_error,
    split_column_parallel,
    split_row_parallel,
)

RNG = np.random.default_rng(20260824)  # fixed seed: reproducible, not cherry-picked


# ============================================================================
# plan_fragments -- the pure allocation algorithm
# ============================================================================

def test_plan_fragments_exact_split():
    workers = [WorkerCapacity("a", 10), WorkerCapacity("b", 10)]
    plan = plan_fragments(20, workers)
    assert plan == {"a": 10, "b": 10}


def test_plan_fragments_never_exceeds_capacity():
    """Heterogeneous, uneven capacities -- no worker's allocation may exceed its cap."""
    workers = [WorkerCapacity("small", 3), WorkerCapacity("mid", 7), WorkerCapacity("big", 40)]
    plan = plan_fragments(37, workers)
    assert plan["small"] <= 3
    assert plan["mid"] <= 7
    assert plan["big"] <= 40
    assert sum(plan.values()) == 37


def test_plan_fragments_rejects_infeasible_pool():
    """The pool genuinely cannot hold the layer -- must raise, never silently under-fit."""
    workers = [WorkerCapacity("a", 2), WorkerCapacity("b", 2)]
    with pytest.raises(ValueError, match="does not fit"):
        plan_fragments(100, workers)


def test_plan_fragments_zero_capacity_worker_gets_nothing():
    """A worker with zero declared capacity (e.g. it can't hold even a fragment of
    this layer) must receive zero units, never a negative or fractional slice."""
    workers = [WorkerCapacity("cannot", 0), WorkerCapacity("can", 50)]
    plan = plan_fragments(50, workers)
    assert plan["cannot"] == 0
    assert plan["can"] == 50


def test_plan_fragments_many_small_heterogeneous_workers():
    """The exact shape this package exists for: MANY small, unequal-capacity workers,
    none able to hold the whole layer, none even holding an equal share."""
    capacities = [3, 5, 2, 9, 4, 7, 1, 6]  # sums to 37, no single worker near that
    workers = [WorkerCapacity(f"w{i}", c) for i, c in enumerate(capacities)]
    plan = plan_fragments(37, workers)
    assert sum(plan.values()) == 37
    for w, cap in zip(workers, capacities):
        assert plan[w.worker_id] <= cap


# ============================================================================
# Column-parallel: split -> execute -> verify against an unsplit reference
# ============================================================================

@pytest.mark.parametrize(
    "out_features,in_features,batch", [(64, 32, 8), (97, 53, 4), (256, 128, 16)]
)
def test_column_parallel_matches_reference(out_features, in_features, batch):
    weight = RNG.standard_normal((out_features, in_features)).astype(np.float64)
    x = RNG.standard_normal((batch, in_features)).astype(np.float64)

    reference = x @ weight.T  # the unsplit computation -- ground truth

    # four HETEROGENEOUS workers, none holding a proportionate 1/4 share -- the exact
    # shape of "consumer GPUs of different sizes, none able to hold the layer alone"
    workers = [
        WorkerCapacity("tiny", max(1, out_features // 12)),
        WorkerCapacity("small", max(1, out_features // 6)),
        WorkerCapacity("mid", max(1, out_features // 3)),
        WorkerCapacity("big", out_features),  # generous cap; actual allocation is proportional
    ]

    fragments = split_column_parallel(weight, workers)
    combined = execute_column_parallel(x, fragments)

    err = relative_error(combined, reference)
    print(
        f"[column-parallel {out_features}x{in_features}, batch={batch}, "
        f"{len(fragments)} fragments] measured relative error: {err:.3e}"
    )
    assert err < 1e-9, f"relative error {err:.3e} exceeds tolerance"
    assert combined.shape == reference.shape


def test_column_parallel_fragment_count_matches_nonzero_allocations():
    """A worker allocated ZERO rows (its capacity share rounded to nothing) must not
    appear as an empty fragment -- confirms the shape, not just the numbers."""
    weight = RNG.standard_normal((10, 4)).astype(np.float64)
    workers = [WorkerCapacity("real", 10), WorkerCapacity("starved", 0)]
    fragments = split_column_parallel(weight, workers)
    assert len(fragments) == 1
    assert fragments[0].worker_id == "real"


# ============================================================================
# Row-parallel: split -> execute -> verify against an unsplit reference
# ============================================================================

@pytest.mark.parametrize(
    "out_features,in_features,batch", [(32, 64, 8), (53, 97, 4), (128, 256, 16)]
)
def test_row_parallel_matches_reference(out_features, in_features, batch):
    weight = RNG.standard_normal((out_features, in_features)).astype(np.float64)
    x = RNG.standard_normal((batch, in_features)).astype(np.float64)

    reference = x @ weight.T

    workers = [
        WorkerCapacity("tiny", max(1, in_features // 12)),
        WorkerCapacity("small", max(1, in_features // 6)),
        WorkerCapacity("mid", max(1, in_features // 3)),
        WorkerCapacity("big", in_features),
    ]

    fragments = split_row_parallel(weight, workers)
    combined = execute_row_parallel(x, fragments)

    err = relative_error(combined, reference)
    print(
        f"[row-parallel {out_features}x{in_features}, batch={batch}, "
        f"{len(fragments)} fragments] measured relative error: {err:.3e}"
    )
    assert err < 1e-9, f"relative error {err:.3e} exceeds tolerance"
    assert combined.shape == reference.shape


# ============================================================================
# Composed: column-parallel up-projection -> row-parallel down-projection, the
# real shape of an MLP block, with NO all-gather between the two stages.
# ============================================================================

def test_composed_mlp_block_no_intermediate_allgather():
    """up-proj (column-parallel) -> activation -> down-proj (row-parallel), across
    the SAME four heterogeneous workers for both stages, with the activation applied
    per-fragment (never on a gathered intermediate) -- this is what proves the two
    splits actually compose, not just that each one independently matches a reference
    in isolation."""
    hidden, intermediate, batch = 64, 256, 8
    x = RNG.standard_normal((batch, hidden)).astype(np.float64)
    w_up = RNG.standard_normal((intermediate, hidden)).astype(np.float64)
    w_down = RNG.standard_normal((hidden, intermediate)).astype(np.float64)

    def relu(a: np.ndarray) -> np.ndarray:
        return np.maximum(a, 0.0)

    # unsplit reference: full up-proj -> relu -> full down-proj
    reference = relu(x @ w_up.T) @ w_down.T

    workers = [
        WorkerCapacity("w0", 20),
        WorkerCapacity("w1", 60),
        WorkerCapacity("w2", 100),
        WorkerCapacity("w3", 76),
    ]  # sums to 256 == intermediate, uneven on purpose

    up_fragments = split_column_parallel(w_up, workers)
    down_fragments = split_row_parallel(w_down, workers)
    # the up-projection's output columns and the down-projection's input columns
    # must align fragment-for-fragment -- assert the plan actually agrees before
    # trusting the composed result, since a silent misalignment would still run and
    # produce a WRONG number rather than an error.
    assert [f.row_start for f in sorted(up_fragments, key=lambda f: f.row_start)] == [
        f.col_start for f in sorted(down_fragments, key=lambda f: f.col_start)
    ]

    # per-worker: local up-proj slice -> local relu -> local down-proj partial
    partial_outputs = []
    for up_f, down_f in zip(
        sorted(up_fragments, key=lambda f: f.row_start),
        sorted(down_fragments, key=lambda f: f.col_start),
    ):
        local_hidden = relu(up_f.forward(x))          # (batch, this worker's slice width)
        partial = local_hidden @ down_f.weight_slice.T  # (batch, hidden) -- already full width
        partial_outputs.append(partial)

    combined = sum(partial_outputs)

    err = relative_error(combined, reference)
    print(f"[composed MLP block, {len(up_fragments)} workers] measured relative error: {err:.3e}")
    assert err < 1e-9, f"relative error {err:.3e} exceeds tolerance"
