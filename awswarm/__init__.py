"""awswarm — sub-layer placement for GPUs too small to hold one full layer.

This package exists because the most mature prior art in this space (shard/c0mpute,
see `.RESEARCH/INTAKE/shard/DOSSIER.md`) explicitly rejects a node that cannot hold at
least one whole transformer layer: "a node that can hold only ~2 layers forces a
62-layer model into ~31 stages... rejected for the ring." That is the right call for
their whole-layer pipeline-parallel design. It is also exactly the boundary this
package exists to push past — consumer GPUs, none able to hold a full layer alone,
jointly computing that layer's output.

HOW THE MODULES FIT TOGETHER
============================
The pieces are a client-shaped substrate an orchestrator can reuse, in this order:

    registry.register(...)            # a volunteer offers capacity — a CLAIM
    registry.verify_and_activate(...) # measured, or it never becomes ACTIVE
    registry.volunteer_workers()      # -> acquire.VolunteerWorker list
    acquire.assess(units, workers)    # will this pool ASSEMBLE? exact probability
    fragment.split_column_parallel()  # cut the layer to each worker's capacity
    protocol.encode(fragment, meta)   # SWARMT01 envelope for the activation traffic
      ... send / receive ...
    protocol.decode(payload)          # checksummed: corruption raises, never decodes
    integrity.aggregate_results(...)  # R replicas agreed? or quarantine a worker
    physical.require_physical(...)    # refuse to RECORD a run as real hardware unless
                                      # the workers were genuinely distinct machines

WHAT IS PROVEN HERE TODAY
=========================
The mechanism, on this machine, with no GPU and no checkpoint: fragment splitting
matches an unsplit reference; assembly probability matches brute-force enumeration over
all 2^N worlds exactly; the codec detects a single flipped bit, a truncated stream and a
reordered chunk; replica disagreement names the offending worker and a split refuses to
pick a winner; a run on one box cannot be labelled PHYSICAL.

WHAT THIS PACKAGE IS NOT
========================
Not a live GPU-rental-fleet controller, and not an orchestrator: nothing here decides
when to reassign work, retry an acquisition, or spend money — that caller is somebody
else's, deliberately. Not a reproduction of any prior run's numbers; the hardware proof
(consumer GPUs jointly executing a real layer fragment) lives outside this repo. Not
validated against a real checkpoint — correctness is shown on tensors, not on Kimi K3 or
Deepseek weights. Availability modelling assumes INDEPENDENCE, which real volunteer
pools violate; every probability here is a ceiling, never a floor. See docs/ROADMAP.md
for the eight gates and an honest status on each.
"""

__version__ = "0.2.0"

from awswarm.acquire import (
    AssemblyReport,
    VolunteerWorker,
    acquisition_probability,
    assess,
    capacity_distribution,
    correlated_ceiling,
    workers_needed,
)
from awswarm.fragment import (
    ColumnFragment,
    RowFragment,
    WorkerCapacity,
    execute_column_parallel,
    execute_row_parallel,
    plan_fragments,
    relative_error,
    split_column_parallel,
    split_row_parallel,
)
from awswarm.integrity import (
    ComputationResult,
    DisagreementDetectedError,
    InsufficientReplicasError,
    IntegrityError,
    IntegrityVerdict,
    IntegrityVote,
    NoQuorumError,
    ReputationManager,
    WorkerReputation,
    aggregate_results,
    is_within_tolerance,
)
from awswarm.physical import (
    ExecutionLabel,
    NotPhysicalError,
    Worker,
    classify,
    require_physical,
)
from awswarm.protocol import (
    BadMagicError,
    ChunkOutOfOrderError,
    ChunkReassembler,
    HeaderChecksumMismatchError,
    PayloadChecksumMismatchError,
    SwarmT01Error,
    TensorMetadata,
    TruncatedStreamError,
    UnknownVersionError,
    decode,
    encode,
    iter_chunks,
)
from awswarm.registry import (
    WorkerCapacityMismatchError,
    WorkerRegistration,
    WorkerRegistry,
    WorkerState,
)

__all__ = [
    "__version__",
    # fragment — cut a layer to fit workers that cannot hold it whole
    "WorkerCapacity", "plan_fragments", "ColumnFragment", "RowFragment",
    "split_column_parallel", "execute_column_parallel",
    "split_row_parallel", "execute_row_parallel", "relative_error",
    # acquire — will the pool actually assemble?
    "VolunteerWorker", "AssemblyReport", "assess", "acquisition_probability",
    "capacity_distribution", "workers_needed", "correlated_ceiling",
    # protocol — SWARMT01 activation transport
    "TensorMetadata", "encode", "decode", "iter_chunks", "ChunkReassembler",
    "SwarmT01Error", "BadMagicError", "UnknownVersionError",
    "HeaderChecksumMismatchError", "PayloadChecksumMismatchError",
    "TruncatedStreamError", "ChunkOutOfOrderError",
    # integrity — did the worker compute the right thing?
    "IntegrityVerdict", "IntegrityVote", "ComputationResult", "WorkerReputation",
    "ReputationManager", "aggregate_results", "is_within_tolerance",
    "IntegrityError", "InsufficientReplicasError", "DisagreementDetectedError",
    "NoQuorumError",
    # registry — who is in the pool, and did they prove it?
    "WorkerState", "WorkerRegistration", "WorkerRegistry",
    "WorkerCapacityMismatchError",
    # physical — is this really distributed hardware?
    "ExecutionLabel", "Worker", "classify", "require_physical", "NotPhysicalError",
]
