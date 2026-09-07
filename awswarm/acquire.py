"""Will this placement actually assemble? — exact acquisition probability for a
pool of volunteer GPUs that individually come and go.

WHY THIS EXISTS
---------------------------------------------------------------------------
`awswarm.fragment` answers "how do I split a layer across workers too small to hold
it". That is half of what this brick's own `adopt` line promises. The other half --
*"plus the probability that plan actually assembles"* -- did not exist, and it is the
half that decides whether volunteer compute is worth attempting at all: a plan across
machines that are each present only sometimes is not a plan until you know how often
all of it is there at once.

The money-losing failure mode this is aimed at is retrying an acquisition that was
never going to complete. The one prior estimate in this lineage (E025 attempt-006,
0.9-2.6%) was produced **ad hoc** and checked in nowhere, so it could not be
re-derived, argued with, or applied to a different pool.

EXACT, NOT SIMULATED
---------------------------------------------------------------------------
Each worker is present independently with probability p_i and contributes c_i units
when present. The assembled capacity is therefore a sum of independent, differently
weighted Bernoullis -- a Poisson-binomial over capacities -- and its full distribution
is computable EXACTLY by convolving one worker at a time. O(N x total_capacity), no
sampling, no seed, same answer every run.

That matters more than speed. A Monte Carlo simulator answers "about 2%" with a
confidence interval nobody propagates, and two runs disagree; the number here is the
number, and `tests/test_acquire.py` checks it against brute-force enumeration over all
2^N subsets for small pools. A claimed probability and a measured one look identical
in a report, which is the whole reason this is arithmetic rather than sampling.

THE ASSUMPTION, STATED PLAINLY
---------------------------------------------------------------------------
Independence is an assumption and it is OPTIMISTIC. Real volunteer pools correlate
hard: a region loses power, a game launches at 8pm and everyone's GPU gets busy, a
driver update lands. Correlated availability makes the true probability LOWER than
this function reports, never higher, so treat the result as a **ceiling** -- useful
for refusing a plan ("even assuming independence this is 3%, do not spend on it") and
not sufficient for promising one. `correlated_ceiling()` below makes that explicit
rather than leaving it in a docstring nobody reads.
"""

from __future__ import annotations

from dataclasses import dataclass

# ============================================================================
# The pool
# ============================================================================


@dataclass(frozen=True)
class VolunteerWorker:
    """One volunteer machine: what it can hold, and how often it is actually there.

    `capacity` is in the same caller-chosen unit as `fragment.WorkerCapacity` (VRAM
    bytes, output rows, whatever) -- this module never interprets it.

    `availability` is the probability the worker is present AND idle at the moment
    you need it. It is not uptime: a machine that is on 24/7 but gaming every evening
    has high uptime and low availability, and the second number is the one that
    decides whether a layer assembles.
    """

    worker_id: str
    capacity: int
    availability: float

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"{self.worker_id}: capacity must be >= 0, got {self.capacity}")
        if not (0.0 <= self.availability <= 1.0):
            raise ValueError(
                f"{self.worker_id}: availability must be in [0, 1], got {self.availability}"
            )


# ============================================================================
# The exact distribution
# ============================================================================


def capacity_distribution(workers: list[VolunteerWorker]) -> list[float]:
    """Exact PMF of total simultaneously-available capacity.

    Returns a list `pmf` where `pmf[k]` is the probability that exactly `k` units of
    capacity are present. Length is `sum(capacity) + 1`, so `pmf[0]` is the
    probability that nothing at all shows up.

    Convolves one worker at a time: after processing a prefix of the pool, `pmf` is
    the exact distribution for that prefix. A zero-capacity worker is folded in
    correctly (it shifts nothing), and so is an always-present one (p=1.0).
    """
    if not workers:
        raise ValueError("no workers given -- an empty pool has no distribution to report")

    total = sum(w.capacity for w in workers)
    pmf = [0.0] * (total + 1)
    pmf[0] = 1.0
    filled = 0

    for w in workers:
        nxt = [0.0] * (total + 1)
        p = w.availability
        for k in range(filled + 1):
            mass = pmf[k]
            if mass == 0.0:
                continue
            nxt[k] += mass * (1.0 - p)          # worker absent
            nxt[k + w.capacity] += mass * p     # worker present
        pmf = nxt
        filled += w.capacity

    return pmf


def acquisition_probability(total_units: int, workers: list[VolunteerWorker]) -> float:
    """P(enough capacity is simultaneously present to hold `total_units`).

    This is the upper tail of `capacity_distribution`. Note it answers a CAPACITY
    question, not a placement one: it assumes any assembled capacity >= the
    requirement can be arranged into a valid split, which is exactly the guarantee
    `fragment.plan_fragments` provides (it raises only when capsum < total_units).
    So the two halves compose: this says whether the pool shows up, and that says
    how to cut the layer once it has.
    """
    if total_units <= 0:
        raise ValueError(f"total_units must be > 0, got {total_units}")
    pmf = capacity_distribution(workers)
    if total_units >= len(pmf):
        return 0.0  # the whole pool, fully present, still cannot hold it
    return sum(pmf[total_units:])


# ============================================================================
# The report a human acts on
# ============================================================================


@dataclass(frozen=True)
class AssemblyReport:
    """What a caller needs to decide whether to spend on an acquisition."""

    total_units: int
    pool_capacity: int
    fits_if_all_present: bool
    probability: float
    expected_capacity: float
    expected_attempts: float
    binding_constraint: str

    def summary(self) -> str:
        pct = self.probability * 100.0
        if not self.fits_if_all_present:
            return (
                f"IMPOSSIBLE: pool holds {self.pool_capacity} units with EVERY worker "
                f"present, and {self.total_units} are needed. No amount of retrying "
                f"fixes this -- {self.binding_constraint}."
            )
        attempts = (
            "never (probability is 0)" if self.expected_attempts == float("inf")
            else f"{self.expected_attempts:.1f} independent attempts on average"
        )
        return (
            f"{pct:.2f}% per attempt; {attempts}. Pool holds {self.pool_capacity} units "
            f"fully present, {self.expected_capacity:.1f} expected; {self.total_units} "
            f"needed. Binding constraint: {self.binding_constraint}. "
            f"CEILING -- assumes independent availability, which real pools violate."
        )


def assess(total_units: int, workers: list[VolunteerWorker]) -> AssemblyReport:
    """Everything needed to accept or refuse a volunteer acquisition, in one object.

    `expected_attempts` is 1/p, the mean of the geometric distribution over
    independent retries -- honest only if the retries really are independent, which
    for a diurnal pool they are not (retrying at 8:05pm after failing at 8:00pm
    samples almost the same world). Reported because a caller who sees "3% per
    attempt, 33 attempts" makes a better decision than one who sees only "3%".
    """
    if total_units <= 0:
        raise ValueError(f"total_units must be > 0, got {total_units}")
    if not workers:
        raise ValueError("no workers given")

    pool = sum(w.capacity for w in workers)
    expected = sum(w.capacity * w.availability for w in workers)
    p = acquisition_probability(total_units, workers) if pool >= total_units else 0.0

    if pool < total_units:
        constraint = (
            f"the pool is {total_units - pool} units short even at full attendance"
        )
    elif expected < total_units:
        constraint = (
            f"expected attendance is {expected:.1f} units, below the {total_units} "
            f"needed -- the pool only clears the bar on a good day"
        )
    else:
        weakest = min(workers, key=lambda w: w.availability)
        constraint = (
            f"expected attendance {expected:.1f} clears {total_units}; the least "
            f"reliable contributor is {weakest.worker_id} at "
            f"{weakest.availability:.0%} holding {weakest.capacity} units"
        )

    return AssemblyReport(
        total_units=total_units,
        pool_capacity=pool,
        fits_if_all_present=pool >= total_units,
        probability=p,
        expected_capacity=expected,
        expected_attempts=(1.0 / p) if p > 0 else float("inf"),
        binding_constraint=constraint,
    )


# ============================================================================
# Planning: how big does the pool need to be?
# ============================================================================


def workers_needed(
    total_units: int,
    capacity: int,
    availability: float,
    target_probability: float,
    max_workers: int = 10_000,
) -> int:
    """Smallest count of identical volunteers reaching `target_probability`.

    The actionable number when recruiting: "at 60% availability and 8GB each, how
    many machines before a 24GB layer assembles 95% of the time?" Grows the pool one
    worker at a time and returns the first count that clears the bar.

    Raises if `max_workers` is reached without clearing it -- an unreachable target
    must fail loudly rather than return a number that quietly does not meet it. That
    is a real case, not a defensive flourish: with availability low enough, adding
    workers raises the probability toward a limit it never crosses in practice, and
    a caller who gets back `max_workers` would read it as an answer.
    """
    if not (0.0 < target_probability < 1.0):
        raise ValueError(
            f"target_probability must be strictly between 0 and 1, got {target_probability}"
        )
    if availability <= 0.0:
        raise ValueError("availability must be > 0 -- a pool that never shows up never assembles")
    if capacity <= 0:
        raise ValueError(f"capacity must be > 0, got {capacity}")

    for n in range(1, max_workers + 1):
        pool = [VolunteerWorker(f"w{i}", capacity, availability) for i in range(n)]
        if acquisition_probability(total_units, pool) >= target_probability:
            return n
    raise ValueError(
        f"{max_workers} workers of {capacity} units at {availability:.0%} availability "
        f"still do not reach {target_probability:.0%} for {total_units} units -- the "
        f"target is not reachable by adding more of this machine"
    )


def correlated_ceiling(probability: float, correlation_note: str = "") -> str:
    """Render a probability as the CEILING it is, so a caller cannot quote it as a floor.

    Exists because the independence assumption is the single most load-bearing thing
    about this module and the easiest to forget once a number is in a slide. Every
    report path routes through wording that carries the caveat with the value.
    """
    tail = f" {correlation_note}" if correlation_note else ""
    return (
        f"<= {probability * 100.0:.2f}% (independence assumed; correlated downtime -- "
        f"evening gaming, regional outages, a driver update -- only lowers it)" + tail
    )
