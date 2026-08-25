# awswarm

Sub-layer fragment placement for GPUs too small to hold one full transformer layer.

## Why

The most mature prior art in heterogeneous-GPU swarm inference (`shard`/c0mpute —
see `.RESEARCH/INTAKE/shard/DOSSIER.md` in the AitherOS monorepo) explicitly rejects
any node that can't hold at least one whole layer: *"a node that can hold only ~2
layers forces a 62-layer model into ~31 stages... rejected for the ring."* That's the
right call for a whole-layer pipeline-parallel design. `awswarm` exists for the case
just past that boundary: four consumer GPUs, none able to hold a full layer alone,
jointly computing that layer's output.

## What's actually here

`awswarm.fragment` — capacity-proportional layer splitting (column-parallel and
row-parallel, the standard tensor-parallel primitives), executed and checked against
an unsplit reference on every test run. CPU-only, numpy-only, deliberately small so
the *mechanism* is independently verifiable on any machine:

```python
from awswarm.fragment import WorkerCapacity, split_column_parallel, execute_column_parallel

workers = [
    WorkerCapacity("tiny-gpu", 8),
    WorkerCapacity("small-gpu", 16),
    WorkerCapacity("mid-gpu", 32),
    WorkerCapacity("big-gpu", 64),
]  # none holds the whole 120-row layer alone
fragments = split_column_parallel(weight, workers)   # weight: (120, in_features)
output = execute_column_parallel(x, fragments)        # matches (x @ weight.T) exactly
```

Run the proof yourself:

```bash
pip install -e .[dev]
pytest tests/ -v -s     # prints the measured relative error on every case, never a claimed one
```

## What's honestly NOT here yet

- No GPU kernels — this is the placement/combine mechanism, not a model runtime.
- No wiring to a real model checkpoint (Kimi K3 or otherwise).
- No live fleet controller, no GPU-rental-marketplace acquisition, no billing.
- Not a reproduction of any specific prior physical GPU run — see `fragment.py`'s
  module docstring for exactly what this package does and does not claim.

## License

Apache-2.0.
