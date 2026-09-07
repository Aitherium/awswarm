"""The acquisition probability must be EXACT, and the test proves it by brute force.

`acquire.capacity_distribution` convolves one worker at a time. The independent check
is enumeration over all 2^N present/absent worlds, summing the probability of each --
a different algorithm, so agreement is evidence rather than a restatement.

Printed, never claimed: every case prints the measured maximum deviation, in the same
spirit as `test_fragment.py` printing relative error.
"""

from __future__ import annotations

import itertools

import pytest
from awswarm.acquire import (
    AssemblyReport,
    VolunteerWorker,
    acquisition_probability,
    assess,
    capacity_distribution,
    correlated_ceiling,
    workers_needed,
)


def brute_force_distribution(workers: list[VolunteerWorker]) -> dict[int, float]:
    """Enumerate every subset. O(2^N) -- only usable for tiny pools, which is the point."""
    dist: dict[int, float] = {}
    for present in itertools.product([False, True], repeat=len(workers)):
        p = 1.0
        cap = 0
        for w, here in zip(workers, present):
            p *= w.availability if here else (1.0 - w.availability)
            if here:
                cap += w.capacity
        dist[cap] = dist.get(cap, 0.0) + p
    return dist


POOLS = [
    [VolunteerWorker("a", 8, 0.9), VolunteerWorker("b", 8, 0.5)],
    [VolunteerWorker("a", 8, 0.6), VolunteerWorker("b", 16, 0.4), VolunteerWorker("c", 4, 0.95)],
    [VolunteerWorker(f"w{i}", 6 + i, 0.3 + 0.1 * i) for i in range(5)],
    # edge shapes that break naive implementations
    [VolunteerWorker("always", 12, 1.0), VolunteerWorker("never", 12, 0.0)],
    [VolunteerWorker("zero-cap", 0, 0.5), VolunteerWorker("real", 10, 0.7)],
]


@pytest.mark.parametrize("workers", POOLS, ids=lambda w: f"{len(w)}-workers")
def test_distribution_matches_brute_force(workers):
    pmf = capacity_distribution(workers)
    bf = brute_force_distribution(workers)
    worst = 0.0
    for cap, prob in bf.items():
        worst = max(worst, abs(pmf[cap] - prob))
    # ...and every capacity the brute force never reaches must carry zero mass
    for cap, prob in enumerate(pmf):
        if cap not in bf:
            worst = max(worst, abs(prob))
    print(f"\n  {len(workers)} workers: max |DP - bruteforce| = {worst:.3e}")
    assert worst < 1e-12
    assert abs(sum(pmf) - 1.0) < 1e-12, "the PMF must sum to 1"


@pytest.mark.parametrize("workers", POOLS, ids=lambda w: f"{len(w)}-workers")
def test_acquisition_probability_matches_brute_force(workers):
    total = sum(w.capacity for w in workers)
    worst = 0.0
    for need in range(1, total + 1):
        got = acquisition_probability(need, workers)
        want = sum(p for cap, p in brute_force_distribution(workers).items() if cap >= need)
        worst = max(worst, abs(got - want))
    print(f"\n  {len(workers)} workers, all thresholds 1..{total}: max dev = {worst:.3e}")
    assert worst < 1e-12


def test_impossible_pool_is_zero_not_an_error():
    """A layer bigger than the whole pool is 0.0, not an exception -- callers ask this
    exact question to DECIDE, and raising would make the answer harder to get than the
    mistake it prevents."""
    pool = [VolunteerWorker("a", 4, 1.0)]
    assert acquisition_probability(99, pool) == 0.0
    r = assess(99, pool)
    assert not r.fits_if_all_present
    assert r.expected_attempts == float("inf")
    assert "IMPOSSIBLE" in r.summary()
    print("\n  " + r.summary())


def test_certain_pool_is_one():
    pool = [VolunteerWorker("a", 10, 1.0), VolunteerWorker("b", 10, 1.0)]
    assert abs(acquisition_probability(20, pool) - 1.0) < 1e-12


def test_assess_reports_a_ceiling_not_a_promise():
    """The independence caveat must travel WITH the number. A probability quoted
    without it gets read as a floor, and correlated downtime only lowers it."""
    pool = [VolunteerWorker(f"w{i}", 8, 0.6) for i in range(4)]
    r = assess(24, pool)
    assert isinstance(r, AssemblyReport)
    assert "CEILING" in r.summary()
    assert "independen" in r.summary().lower()
    print(f"\n  {r.summary()}")


def test_workers_needed_is_the_smallest_that_clears():
    """It must return the FIRST count over the bar -- one fewer must miss it, or the
    answer is padded and a recruiter over-provisions on our advice."""
    need, cap, avail, target = 24, 8, 0.6, 0.95
    n = workers_needed(need, cap, avail, target)
    at_n = acquisition_probability(need, [VolunteerWorker(f"w{i}", cap, avail) for i in range(n)])
    at_n_1 = acquisition_probability(
        need, [VolunteerWorker(f"w{i}", cap, avail) for i in range(n - 1)]
    )
    print(f"\n  {need} units at {cap}/{avail:.0%}: {n} workers -> {at_n:.4f}, "
          f"{n - 1} -> {at_n_1:.4f} (target {target})")
    assert at_n >= target
    assert at_n_1 < target


def test_unreachable_target_raises_rather_than_returning_max():
    """Returning max_workers would read as an answer that meets the target."""
    with pytest.raises(ValueError, match="not reachable"):
        workers_needed(100, 8, 0.05, 0.99, max_workers=6)


def test_correlated_ceiling_carries_the_caveat():
    s = correlated_ceiling(0.031)
    assert "3.10%" in s and "<=" in s and "correlated" in s


def test_low_availability_pool_is_honest_about_being_hopeless():
    """The shape this module was built for: enough capacity on paper, almost never
    present at once. A capacity-only view calls this fine; the probability does not."""
    pool = [VolunteerWorker(f"v{i}", 8, 0.25) for i in range(6)]
    r = assess(40, pool)
    print(f"\n  {r.summary()}")
    assert r.fits_if_all_present          # 48 units on paper vs 40 needed
    assert r.probability < 0.05           # ...and it essentially never happens
    assert r.expected_attempts > 20
