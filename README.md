# awswarm

Sub-layer fragment placement for GPUs too small to hold one full transformer layer —
plus the parts that decide whether a pool of them is worth trying at all.

## Why

The most mature prior art in heterogeneous-GPU swarm inference (`shard`/c0mpute —
see `.RESEARCH/INTAKE/shard/DOSSIER.md` in the AitherOS monorepo) explicitly rejects
any node that can't hold at least one whole layer: *"a node that can hold only ~2
layers forces a 62-layer model into ~31 stages... rejected for the ring."* That's the
right call for a whole-layer pipeline-parallel design. `awswarm` exists for the case
just past that boundary: consumer GPUs, none able to hold a full layer alone, that can
be split proportional to capacity so they jointly compute that layer's output.

The splitting math is proven here. **A prior physical run of this shape exists but its
data was not recovered, so nothing in this repo reproduces it** — see
`docs/ROADMAP.md` for gate-by-gate status.

## How the modules fit together

Six pieces, each usable alone, in the order an orchestrator would call them:

```python
from awswarm import (
    WorkerRegistry, assess, split_column_parallel, encode, decode,
    aggregate_results, require_physical,
)

# 1. Who is in the pool — and did they PROVE it?  A declared capacity is a claim.
reg = WorkerRegistry()
reg.register("volunteer-1", capacity=8, availability=0.6)   # UNVERIFIED
reg.verify_and_activate("volunteer-1", measured_capacity=8) # ACTIVE, or it never joins

# 2. Will enough of them be present AT ONCE?  Exact, not simulated.
report = assess(total_units=24, workers=reg.volunteer_workers())
if report.probability < 0.5:
    return report.summary()      # refuse before spending on an acquisition

# 3. Cut the layer to each worker's capacity.
fragments = split_column_parallel(weight, capacities)

# 4. Ship the activation traffic.  Corruption RAISES; it never decodes to garbage.
payload = encode(fragment_tensor, meta)
tensor, meta = decode(payload)

# 5. Did the worker compute the right thing?  Identity is not integrity.
vote = aggregate_results(replica_results, tolerance=1e-4)

# 6. Refuse to RECORD the run as real hardware unless it was.
require_physical(workers, controller_address)
```

A runnable version of exactly this is `examples/end_to_end.py`.

## What's actually here

| module | answers |
|---|---|
| `fragment` | how do I split a layer across workers too small to hold it? |
| `acquire` | will this pool ever be present at once? — exact probability |
| `protocol` | SWARMT01 envelope for the activation traffic, checksummed and chunked |
| `integrity` | did the worker compute the right thing? — replica consensus |
| `registry` | who is in the pool, and was their capacity measured or claimed? |
| `physical` | is this really distributed hardware, or one box wearing hats? |

Run the proof yourself — every number is measured on the run, never quoted:

```bash
pip install -e .[dev]
pytest -q -s          # 173 tests; prints measured deviations, not claimed ones
```

Two results worth knowing before you build on this:

- **Assembly probability is exact.** It convolves the per-worker availability
  distribution rather than sampling it, and the tests check it against brute-force
  enumeration of all 2^N worlds — measured deviation `0.000e+00`.
- **Capacity on paper is not capacity.** Six 8-unit volunteers at 25% availability hold
  48 units and assemble a 40-unit layer **0.46%** of the time — 215 attempts on average.
  A capacity-only view calls that pool fine.

## What's honestly NOT here

- **No orchestrator.** Nothing here decides when to reassign work, retry an
  acquisition, or spend money. That caller is deliberately somebody else's — these are
  the primitives it would reuse, not a controller.
- **No GPU kernels, no model runtime.** Correctness is shown on tensors, not against a
  real checkpoint (Kimi K3, Deepseek, or otherwise). Gates 1 and 3 in the roadmap are
  PARTIAL for exactly this reason.
- **No live pool data.** Availability, churn and correlation-of-failures have never been
  measured on a real volunteer pool. Every probability assumes **independence**, which
  real pools violate — evening gaming, regional outages, a driver update. That makes
  every number here a **ceiling, never a floor**: useful for refusing a plan, not
  sufficient for promising one.
- **No network cost accounting.** The protocol proves byte-fidelity; it does not measure
  transport latency or compare it against the compute gain.
- **No reproduction of any prior physical run.** The hardware proof lives outside this
  repository.

Volunteer compute is closer in *substrate*, not in deployment readiness. This package
is the reusable, honest foundation an orchestrator can stand on.

## License

Apache-2.0. The SWARMT01 wire format is adapted from `swarm-inference-lab`
(Apache-2.0, 2026) with attribution recorded in `awswarm/protocol.py`.
