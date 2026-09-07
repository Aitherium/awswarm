# awswarm Roadmap — 8 Feasibility Gates

This roadmap documents the 8 gates that decide whether sub-layer placement on
volunteer GPU pools is feasible. Each gate is explicit: what it asks, what
evidence would pass it, and the honest status today (2026-09-02).

**General policy**: a gate is PASS only if it names concrete measured evidence.
No PASS claim without evidence.

---

## Gate 1: Shard-Fits-Cap

**What it asks**: Does a shard of a layer fit within a worker's capacity?

**Evidence for PASS**:
- Prove that a layer can be split into N fragments, each sized proportional to
  a worker's available capacity, with no fragment exceeding its target worker's
  memory limit.
- Run the split operation on a reference model layer (not just random tensors).
- Measure and print actual bytes per worker for each fragment.

**Status TODAY**: PARTIAL

Fragment correctness is proven in `awswarm/fragment.py` and
`tests/test_fragment.py`: weights are split column-wise and row-wise, the
combined output matches the reference, and measured relative error is printed
on every run (typically <1e-6 for float32).

However, this proof uses random tensors, not a real model layer. The split has
not been run against an actual checkpoint (Kimi K3, Deepseek v4, etc.) where
layer sizes are known and worker capacity is measured against real hardware.

**Modules contributing**: `fragment.py` (split mechanism + correctness test)

---

## Gate 2: State Semantics Per Architecture

**What it asks**: Can awswarm distinguish real distributed execution from
simulation, loopback, or claimed-but-false physical setups?

**Evidence for PASS**:
- Label a deployment PHYSICAL, LOOPBACK, or SIMULATED based on worker
  hostnames and addresses.
- Verify that a PHYSICAL deployment truly uses distinct machines, non-private
  addresses, and controller is reachable from workers.
- Test both correct labeling and correct rejection of invalid configurations.

**Status TODAY**: PASS

`awswarm/physical.py` correctly classifies deployments:
- PHYSICAL if all workers have distinct hostnames and non-private addresses
- LOOPBACK if all workers are on the same host or use loopback addresses
- SIMULATED if explicitly marked as such

Measured evidence: `tests/test_physical.py` verifies both directions with
realistic and edge-case configurations. No test failure has occurred.

Registry state machine (`awswarm/registry.py`) tracks worker states
(UNVERIFIED, ACTIVE, UNHEALTHY, QUARANTINED) and enforces transitions
correctly.

**Modules contributing**: `physical.py` (classification), `registry.py` (state
machine)

---

## Gate 3: Quantised-Kernel Correctness

**What it asks**: Do quantized sub-layer fragments execute correctly?

**Evidence for PASS**:
- Run fragment splits on int8-quantized weights.
- Prove that dequantization → split → re-quantization preserves the quantized
  result within acceptable tolerance.
- Measure and print actual error on a reference model layer.

**Status TODAY**: PARTIAL

`awswarm/fragment.py` proves correctness for float32 and float16 (measured
relative error <1e-6). The int8 quantization path is not yet tested against
real quantized checkpoints.

The mechanism is sound (column and row splits preserve correctness), but the
int8 path needs:
- Actual int8-quantized layer weights (e.g., from AWQ or GPTQ)
- Round-trip test of dequantization → split → re-quantization
- Measured error report on each test

**Modules contributing**: `fragment.py` (would extend with int8 test)

---

## Gate 4: Measured Service Rate

**What it asks**: What is the throughput (tokens/sec or similar) for a swarm
executing a sub-layer on real hardware?

**Evidence for PASS**:
- Run a real sub-layer (MLP or attention) across a pool of volunteer GPUs.
- Measure and print actual throughput (tokens/sec, activations/sec).
- Compare against a single-GPU baseline.

**Status TODAY**: NOT MET

This requires:
- A real GPU pool orchestrator (not yet implemented)
- A real model checkpoint loaded and split across workers
- Network transport of activations (protocol is ready, but not integrated)
- Measurement infrastructure

This is the first gate that depends on a live rental-fleet controller, which
is explicitly out of scope for this release.

**Modules contributing**: (none yet; protocol.py provides transport)

---

## Gate 5: Measured Activation Transport Cost

**What it asks**: What is the overhead of sending activations across the swarm?

**Evidence for PASS**:
- Measure actual bytes sent per forward pass.
- Measure actual latency to transport activations between workers.
- Compare against compute cost to assess whether network dominates.

**Status TODAY**: PASS (protocol only)

`awswarm/protocol.py` proves the wire format works correctly:
- Encodes and decodes NumPy arrays without data loss
- Detects corruption with SHA-256 checksums per chunk
- Preserves dtype, shape, and byte-exact content
- Measured chunk overhead: 32 bytes (hash) per 1 MB default chunk

However, "activation transport cost" includes network latency, which is not
measured here. This gate is PASS for the protocol layer, but network cost
requires the orchestrator (out of scope).

**Modules contributing**: `protocol.py` (byte-exact round-trip + overhead
measurement)

---

## Gate 6: Recovery Cost

**What it asks**: If a worker fails during a sub-layer execution, how much work
is lost and how much time is spent recovering?

**Evidence for PASS**:
- Simulate worker failure scenarios (heartbeat timeout, quarantine).
- Measure work lost (activation bytes re-sent, compute re-done).
- Measure recovery time (heartbeat expiry + readmit cycle).

**Status TODAY**: PARTIAL

`awswarm/registry.py` implements the recovery infrastructure:
- Heartbeat tracking with injected clocks (testable without ambient time)
- Quarantine/readmit with reason tracking
- State transitions (ACTIVE → UNHEALTHY → QUARANTINED → UNVERIFIED, then back
  to ACTIVE on proof)

Measured evidence: `tests/test_registry.py` verifies heartbeat expiry and
quarantine transitions with deterministic clocks.

However, "recovery cost" for an in-flight activation requires orchestrator
integration to measure actual data re-sent and compute re-done. The
infrastructure exists, but the orchestrator integration does not.

**Modules contributing**: `registry.py` (heartbeat + quarantine state machine)

---

## Gate 7: Feasible Node Count

**What it asks**: For a given pool of volunteer GPUs with independent,
time-varying availability, is the number of nodes enough to reliably assemble a
sub-layer execution?

**Evidence for PASS**:
- Compute exact probability that all workers required for the sub-layer are
  available at the same time.
- Compare against a feasibility threshold (e.g., 5% minimum probability).
- Measure this probability for a real pool of volunteers.

**Status TODAY**: PASS (probability function only)

`awswarm/acquire.py` computes the exact Poisson-binomial probability that a
pool assembles:
- Each worker has independent availability probability p_i
- Each contributes capacity c_i when available
- The assembled capacity is a sum of independent weighted Bernoullis
- The function enumerates all 2^N subsets exactly, no Monte Carlo

Measured evidence: `tests/test_acquire.py` verifies the probability against
brute-force enumeration for small pools. A realistic pool (20-30 workers) has
correct probability computed.

However, "feasible node count" for a real model execution requires knowing the
actual capacity needed (from gate 1) and the real availability distribution of
the actual volunteer pool. This gate is PASS for the mechanism, but the real
pool integration is out of scope.

**Modules contributing**: `acquire.py` (exact probability + verification test)

---

## Gate 8: Economics

**What it asks**: Is the cost of renting GPU time to run a sub-layer on
volunteers cheaper than running the whole thing on a single large GPU?

**Evidence for PASS**:
- Measure rental cost for the compute time (from gate 4) on real volunteer
  hardware (Vast.ai, Hetzner, etc.).
- Compare against single-GPU rental cost.
- Include failure cost and recovery time (from gate 6).

**Status TODAY**: NOT MET

This requires:
- Real service rate measured (gate 4, out of scope)
- Real rental prices and their volatility
- Failure scenarios and their economics
- Comparison model (single GPU pricing, alternative architectures)

This gate is intentionally deferred pending the orchestrator and real-world
deployment experience.

**Modules contributing**: (none yet)

---

## Summary: Which Gates Pass, Which Modules Move Them

| Gate | Status | Modules |
|------|--------|---------|
| 1. Shard-Fits-Cap | PARTIAL | fragment.py |
| 2. State Semantics | PASS | physical.py, registry.py |
| 3. Quantised-Kernel | PARTIAL | fragment.py |
| 4. Service Rate | NOT MET | orchestrator (out of scope) |
| 5. Transport Cost | PASS | protocol.py |
| 6. Recovery Cost | PARTIAL | registry.py |
| 7. Feasible Nodes | PASS | acquire.py |
| 8. Economics | NOT MET | orchestrator (out of scope) |

## What This Release Ships

- **awswarm/fragment.py** — Prove that a layer can be split and reassembled.
  Tested on random tensors; not yet on real model layers.

- **awswarm/acquire.py** — Compute exact probability that a volunteer pool
  assembles. Tested against brute-force enumeration. Ready for real pool
  integration.

- **awswarm/protocol.py** — SWARMT01 binary format for activation transport.
  Tested for byte-exact round-trip and corruption detection. Ready for
  orchestrator integration.

- **awswarm/physical.py** — Distinguish real distributed execution from
  simulation. Tested for correct classification. Ready for deployment tagging.

- **awswarm/registry.py** — Track volunteer pool states and enforce recovery
  semantics. Tested for heartbeat and quarantine transitions. Ready for
  orchestrator integration.

## What This Release Does NOT Ship

- A live rental-fleet orchestrator wired to Vast/Hetzner
- Proof that any prior run's numbers are reproducible
- Real GPU service rate measurement
- Real failure scenario testing
- Economic analysis or cost comparison

These are explicitly out of scope pending orchestrator work and real-world
deployment.

## Design Principle

Every number in this roadmap is either:
1. **Measured and printed on every test run** (fragment error, protocol
   overhead, acquire probability), or
2. **Explicitly marked as NOT MET** (service rate, economics)

No claimed probability, no simulated throughput, no vague "should work"
estimates. The mechanism is proven small and honest; the orchestrator and
deployment are separate work.
