"""The whole package in one flow — registry → acquire → fragment → protocol → integrity
→ physical.

WHY THIS IS A TEST AND NOT A README SNIPPET
---------------------------------------------------------------------------
The completeness critic's sharpest finding was that six modules existed and nothing
showed them talking to each other: "no example showing fragment → protocol → network →
integrity → registry flow". An example in prose rots the first time a signature moves
and nobody notices, because prose is not executed. This one runs on every suite.

It is also the only place the SEAMS are exercised. Each module's own tests prove that
module; a seam belongs to neither side, which is exactly why seams are where the
integration defects live. Writing this found a real one: `WorkerRegistry` had no way to
hand its pool to `acquire.assess()` at all, so a caller had to hand-translate and would
have reached for the DECLARED capacity — the number a volunteer claimed about hardware
nobody measured. `volunteer_workers()` exists because of this file.

What it deliberately does NOT do: simulate a network, orchestrate retries, or decide
anything. There is no orchestrator here and this does not pretend to be one — it shows
the primitives composing, which is the honest claim.
"""

from __future__ import annotations

import numpy as np
import pytest
from awswarm import (
    ComputationResult,
    ExecutionLabel,
    IntegrityVerdict,
    NotPhysicalError,
    TensorMetadata,
    Worker,
    WorkerCapacity,
    WorkerRegistry,
    aggregate_results,
    assess,
    classify,
    decode,
    encode,
    execute_column_parallel,
    require_physical,
    split_column_parallel,
)

CONTROLLER = "9.9.9.9"


def test_full_volunteer_flow_end_to_end():
    """One layer, one volunteer pool, every seam crossed once."""

    # ---- 1. the pool: a claim, then a measurement -------------------------------
    reg = WorkerRegistry()
    declared = [("volunteer-a", 32, 0.85), ("volunteer-b", 16, 0.70),
                ("volunteer-c", 8, 0.60), ("volunteer-d", 8, 0.55)]
    for wid, cap, avail in declared:
        reg.register(wid, capacity=cap, availability=avail)
    assert not reg.available_workers(), "UNVERIFIED workers must not be placeable"

    for wid, cap, _ in declared:
        reg.verify_and_activate(wid, measured_capacity=cap)
    pool = reg.volunteer_workers()
    print(f"\n  1. pool: {len(pool)} verified volunteers, "
          f"{sum(w.capacity for w in pool)} units at full attendance")

    # ---- 2. will it assemble? ----------------------------------------------------
    need = 48
    report = assess(need, pool)
    print(f"  2. assemble {need} units: {report.probability:.2%} per attempt, "
          f"{report.expected_attempts:.1f} attempts")
    assert report.fits_if_all_present
    assert "CEILING" in report.summary(), "the independence caveat must travel with it"

    # ---- 3. cut the layer to measured capacity -----------------------------------
    rows, in_features = 64, 12
    rng = np.random.default_rng(1234)
    weight = rng.standard_normal((rows, in_features)).astype(np.float32)
    x = rng.standard_normal((3, in_features)).astype(np.float32)

    caps = [WorkerCapacity(w.worker_id, w.capacity) for w in pool]
    fragments = split_column_parallel(weight, caps)
    assert sum(f.weight_slice.shape[0] for f in fragments) == rows, "the split must be lossless"
    print(f"  3. split {rows} rows across {len(fragments)} workers, none holding it alone")

    # ---- 4. the activation traffic survives the wire ------------------------------
    shipped = []
    for f in fragments:
        payload = encode(f.weight_slice,
                         TensorMetadata(str(f.weight_slice.dtype), f.weight_slice.shape))
        tensor, meta = decode(payload)
        assert np.array_equal(tensor, f.weight_slice), "the wire must not alter the tensor"
        assert meta.shape == tuple(f.weight_slice.shape)
        shipped.append(len(payload))
    print(f"  4. {len(shipped)} fragments encoded+decoded byte-exact "
          f"({sum(shipped):,} bytes on the wire)")

    # ---- 5. the combined result matches an unsplit reference ----------------------
    combined = execute_column_parallel(x, fragments)
    reference = x @ weight.T
    dev = float(np.max(np.abs(combined - reference)))
    print(f"  5. combined vs unsplit reference: max deviation {dev:.3e}")
    assert dev < 1e-4

    # ---- 6. did the workers compute the right thing? ------------------------------
    honest = [ComputationResult(tensor=combined.copy(), worker_id=f"replica-{i}")
              for i in range(3)]
    vote = aggregate_results(honest, tolerance=1e-5)
    assert vote.verdict is IntegrityVerdict.AGREEMENT
    print(f"  6. 3 honest replicas -> {vote.verdict.value}, "
          f"max deviation {vote.max_deviation:.3e}")

    # ---- 7. is this really distributed hardware? ----------------------------------
    real = [Worker("gpu-a", "8.8.8.8"), Worker("gpu-b", "1.1.1.1"),
            Worker("gpu-c", "8.8.4.4"), Worker("gpu-d", "208.67.222.222")]
    require_physical(real, CONTROLLER)
    assert classify(real, CONTROLLER) is ExecutionLabel.PHYSICAL
    print("  7. four distinct public workers -> PHYSICAL (recorded honestly)")


def test_the_flow_refuses_at_every_gate_it_should():
    """The same pipeline, each gate given something it must reject.

    Paired with the happy path above on purpose: a pipeline that only ever succeeds in
    its example is a demo, and a demo tells you nothing about the case that matters.
    """
    reg = WorkerRegistry()

    # a volunteer whose hardware does not match its claim never becomes placeable
    reg.register("liar", capacity=64, availability=0.9)
    with pytest.raises(Exception) as e:
        reg.verify_and_activate("liar", measured_capacity=8)
    print(f"\n  registry refuses the claim: {str(e.value)[:80]}")
    assert not reg.available_workers()

    # a pool that cannot hold the layer says so instead of returning a small number
    reg2 = WorkerRegistry()
    reg2.register("tiny", capacity=4, availability=0.9)
    reg2.verify_and_activate("tiny", measured_capacity=4)
    impossible = assess(4096, reg2.volunteer_workers())
    assert impossible.probability == 0.0
    assert "IMPOSSIBLE" in impossible.summary()
    print(f"  acquire refuses: {impossible.summary()[:80]}")

    # a corrupted replica is named, not averaged away
    base = np.ones((4, 4), dtype=np.float32)
    results = [
        ComputationResult(tensor=base.copy(), worker_id="honest-1"),
        ComputationResult(tensor=base.copy(), worker_id="honest-2"),
        ComputationResult(tensor=base + 0.5, worker_id="corrupted"),
    ]
    with pytest.raises(Exception) as e:
        aggregate_results(results, tolerance=1e-3)
    assert "corrupted" in str(e.value)
    print(f"  integrity names the offender: {str(e.value)[:80]}")

    # and a run on one box cannot be recorded as hardware
    with pytest.raises(NotPhysicalError):
        require_physical([Worker("box", "127.0.0.1")], "127.0.0.1")
    print("  physical refuses to record a loopback run as real hardware")
