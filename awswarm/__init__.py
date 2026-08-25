"""awswarm — sub-layer placement for GPUs too small to hold one full layer.

This package exists because the most mature prior art in this space (shard/c0mpute,
see `.RESEARCH/INTAKE/shard/DOSSIER.md`) explicitly rejects a node that cannot hold at
least one whole transformer layer: "a node that can hold only ~2 layers forces a
62-layer model into ~31 stages... rejected for the ring." That is the right call for
their whole-layer pipeline-parallel design. It is also exactly the boundary this
package exists to push past — four consumer GPUs, none able to hold a full layer
alone, jointly computing that layer's output.

`fragment.py` is the mechanism: split ONE layer's weight matrix across N workers of
heterogeneous capacity, execute each fragment, combine the partial results, and prove
the combined result matches an unsplit reference within numerical tolerance. It is
deliberately small, CPU-only, and dependency-light (numpy only) so the MECHANISM is
verifiable on any machine, independent of what GPU hardware or model checkpoint is
available.

What this package is NOT (yet): a live multi-GPU-rental-fleet controller wired to
Vast/Hetzner accounts and billing, or a proof this reproduces any specific prior run's
numbers. See `fragment.py`'s module docstring for exactly what is proven here today.
"""

__version__ = "0.1.0"
