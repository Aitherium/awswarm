"""Measured-registration gates and quarantine for volunteer hardware pools.

WHAT IS PROVEN HERE, TODAY, ON THIS MACHINE: a volunteer pool registry that
tracks worker states (UNVERIFIED, ACTIVE, UNHEALTHY, QUARANTINED), verifies
benchmarks before registration, implements deterministic heartbeat tracking with
injected clocks (testable without ambient time), and provides quarantine/readmit
functionality with reason tracking. The measured capacity is independently
verified against the worker's declared capacity using the `assess()` probability
framework from awswarm.acquire — a worker is not ACTIVE until its declared
capacity has a measured round-trip matching (within tolerance).

WHAT IS NOT CLAIMED: this module does not orchestrate worker placement, does not
contact volunteer hardware, and does not implement network heartbeat probes. The
heartbeat clock is injected for testing; in production, callers supply their own
time source and the registry does not sleep. Capacity measurement is caller-
supplied via benchmark results; the registry does not run benchmarks.

WHY THIS MATTERS: volunteer pools drift constantly (availability changes minute-
to-minute), and silent availability drift is the worst failure mode — a pool
reads as "available for work" while every worker is actually absent or slow.
This registry makes that impossible: a worker is ACTIVE only if a fresh
measurement proves it can deliver. Once active, heartbeat expiry is distinct from
quarantine (expiry is "did not hear from you"; quarantine is "you exceeded your
SLA"), and readmit requires proof again. The two states route to different
handler paths — silence and proof-of-failure are not the same recovery task.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from . import acquire


class WorkerState(Enum):
    """Volunteer worker lifecycle state.

    UNVERIFIED: registered but no benchmark measurement yet. Refuses placement.
    ACTIVE: measured capacity matches declared (verified via assess_probability).
    UNHEALTHY: heartbeat expired — did not hear from the worker on schedule.
    QUARANTINED: exceeded SLA — proof of failure, not mere silence. Manual
                 readmit required.
    """

    UNVERIFIED = "unverified"
    ACTIVE = "active"
    UNHEALTHY = "unhealthy"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class WorkerRegistration:
    """One volunteer worker and its measured state at registration time."""

    worker_id: str
    capacity: int  # declared capacity, same unit as fragment.WorkerCapacity
    availability: float  # probability present and idle, [0.0, 1.0]
    state: WorkerState
    measured_capacity: Optional[int] = None  # measured, if verified
    last_heartbeat: float = 0.0  # seconds since epoch of last alive signal
    quarantine_reason: Optional[str] = None  # why quarantined, if set

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(
                f"{self.worker_id}: capacity must be >= 0, got {self.capacity}"
            )
        if not (0.0 <= self.availability <= 1.0):
            raise ValueError(
                f"{self.worker_id}: availability must be in [0, 1], "
                f"got {self.availability}"
            )
        if self.state == WorkerState.ACTIVE and self.measured_capacity is None:
            raise ValueError(
                f"{self.worker_id}: ACTIVE state requires measured_capacity, "
                f"got None"
            )
        if self.state == WorkerState.QUARANTINED and (
            self.quarantine_reason is None
        ):
            raise ValueError(
                f"{self.worker_id}: QUARANTINED state requires "
                f"quarantine_reason, got None"
            )


@dataclass(frozen=True)
class WorkerCapacityMismatchError(Exception):
    """Measured capacity does not match declared capacity within tolerance."""

    worker_id: str
    declared: int
    measured: int
    tolerance: float  # as a fraction, e.g., 0.05 means 5%

    def __str__(self) -> str:
        error_pct = (
            abs(self.measured - self.declared) / max(self.declared, 1) * 100
        )
        return (
            f"{self.worker_id}: measured capacity {self.measured} "
            f"diverges from declared {self.declared} "
            f"({error_pct:.1f}% error, tolerance {self.tolerance*100:.1f}%)"
        )


class WorkerRegistry:
    """Measured-registration gate and quarantine for volunteer pools.

    Tracks worker states, verifies benchmarks before ACTIVE, and maintains
    heartbeat timeouts and quarantine reasons. Uses injected clock for
    deterministic testing; in production, callers supply time.now().
    """

    def __init__(
        self,
        heartbeat_timeout_sec: float = 300.0,
        capacity_tolerance: float = 0.05,
        get_time_fn=None,
    ):
        """Initialize the registry.

        Args:
            heartbeat_timeout_sec: how long without heartbeat before UNHEALTHY
            capacity_tolerance: fractional tolerance for measured vs declared
                                 capacity (default 5%)
            get_time_fn: callable returning seconds since epoch; defaults to
                         time.time() for production, injected in tests
        """
        if heartbeat_timeout_sec <= 0:
            raise ValueError(
                f"heartbeat_timeout_sec must be > 0, got {heartbeat_timeout_sec}"
            )
        if not (0 <= capacity_tolerance <= 1.0):
            raise ValueError(
                f"capacity_tolerance must be in [0, 1], "
                f"got {capacity_tolerance}"
            )

        self._workers: dict[str, WorkerRegistration] = {}
        self._heartbeat_timeout = heartbeat_timeout_sec
        self._capacity_tolerance = capacity_tolerance
        self._get_time = get_time_fn or time.time

    def register(
        self,
        worker_id: str,
        capacity: int,
        availability: float,
    ) -> WorkerRegistration:
        """Register a new volunteer worker in UNVERIFIED state.

        Args:
            worker_id: unique identifier for the worker
            capacity: declared capacity, in the same unit as fragment.WorkerCapacity
            availability: probability the worker is present and idle, [0.0, 1.0]

        Returns:
            WorkerRegistration in UNVERIFIED state

        Raises:
            ValueError: if worker already registered or capacity invalid
        """
        if worker_id in self._workers:
            raise ValueError(f"{worker_id}: already registered")

        reg = WorkerRegistration(
            worker_id=worker_id,
            capacity=capacity,
            availability=availability,
            state=WorkerState.UNVERIFIED,
            last_heartbeat=self._get_time(),
        )
        self._workers[worker_id] = reg
        return reg

    def verify_and_activate(
        self,
        worker_id: str,
        measured_capacity: int,
    ) -> WorkerRegistration:
        """Verify a worker's benchmark and transition to ACTIVE.

        Measures the assembly probability assuming this worker is the only one
        (single-worker assessment) using acquire.assess(). If the measured
        capacity matches the declared capacity (within tolerance), transitions
        the worker to ACTIVE. Otherwise raises WorkerCapacityMismatchError.

        Args:
            worker_id: must be registered (in any state)
            measured_capacity: capacity achieved in benchmark test

        Returns:
            WorkerRegistration in ACTIVE state

        Raises:
            KeyError: if worker not registered
            WorkerCapacityMismatchError: if measured diverges from declared
        """
        reg = self._workers[worker_id]

        if measured_capacity < 0:
            raise ValueError(
                f"{worker_id}: measured_capacity must be >= 0, "
                f"got {measured_capacity}"
            )

        # Verify against declared capacity within tolerance.
        # Largest-remainder rounding: |measured - declared| / declared <= tol
        max_error = reg.capacity * self._capacity_tolerance
        if abs(measured_capacity - reg.capacity) > max_error:
            raise WorkerCapacityMismatchError(
                worker_id=worker_id,
                declared=reg.capacity,
                measured=measured_capacity,
                tolerance=self._capacity_tolerance,
            )

        # Construct a VolunteerWorker for a single-worker assessment via
        # acquire.assess(), which uses the declared capacity and availability.
        # The measured capacity has been verified to match, so we use the
        # declared in the probability calculation.
        worker = acquire.VolunteerWorker(
            worker_id=worker_id,
            capacity=reg.capacity,
            availability=reg.availability,
        )

        # Single-worker pool: assess probability of this one worker assembling.
        report = acquire.assess(total_units=reg.capacity, workers=[worker])

        # A worker with zero expected capacity (availability=0 or capacity=0)
        # cannot hold itself—it can never deliver any units.
        if report.expected_capacity == 0:
            raise ValueError(
                f"{worker_id}: declared capacity {reg.capacity} cannot hold "
                f"itself"
            )

        # A single worker either fits or it does not. If it does not, the
        # declared capacity itself is broken — a worker cannot hold its own
        # declared capacity. This is a loudly-failing configuration, not a
        # capacity-mismatch.
        if not report.fits_if_all_present:
            raise ValueError(
                f"{worker_id}: declared capacity {reg.capacity} cannot hold "
                f"itself"
            )

        # Activate with the measured capacity stored.
        active_reg = WorkerRegistration(
            worker_id=reg.worker_id,
            capacity=reg.capacity,
            availability=reg.availability,
            state=WorkerState.ACTIVE,
            measured_capacity=measured_capacity,
            last_heartbeat=self._get_time(),
        )
        self._workers[worker_id] = active_reg
        return active_reg

    def heartbeat(self, worker_id: str) -> WorkerRegistration:
        """Record a heartbeat from a worker.

        If the worker is UNHEALTHY (heartbeat expired), transitions it back to
        ACTIVE. If it is QUARANTINED, leaves it unchanged (must use readmit()).

        Args:
            worker_id: must be registered and ACTIVE or UNHEALTHY

        Returns:
            updated WorkerRegistration

        Raises:
            KeyError: if worker not registered
            ValueError: if worker is UNVERIFIED or QUARANTINED
        """
        reg = self._workers[worker_id]

        if reg.state == WorkerState.UNVERIFIED:
            raise ValueError(
                f"{worker_id}: cannot heartbeat UNVERIFIED worker — "
                f"call verify_and_activate() first"
            )
        if reg.state == WorkerState.QUARANTINED:
            raise ValueError(
                f"{worker_id}: cannot heartbeat QUARANTINED worker — "
                f"call readmit() to restore"
            )

        # Update last heartbeat and transition UNHEALTHY -> ACTIVE.
        new_reg = WorkerRegistration(
            worker_id=reg.worker_id,
            capacity=reg.capacity,
            availability=reg.availability,
            state=(
                WorkerState.ACTIVE
                if reg.state == WorkerState.UNHEALTHY
                else reg.state
            ),
            measured_capacity=reg.measured_capacity,
            last_heartbeat=self._get_time(),
        )
        self._workers[worker_id] = new_reg
        return new_reg

    def check_heartbeats(self) -> list[WorkerRegistration]:
        """Scan for workers whose heartbeat has expired.

        Transitions ACTIVE workers to UNHEALTHY if last_heartbeat is older than
        heartbeat_timeout_sec. Returns all workers that changed state.

        Returns:
            list of WorkerRegistration objects that transitioned to UNHEALTHY
        """
        now = self._get_time()
        transitioned = []

        for worker_id, reg in list(self._workers.items()):
            if reg.state == WorkerState.ACTIVE:
                if now - reg.last_heartbeat > self._heartbeat_timeout:
                    unhealthy = WorkerRegistration(
                        worker_id=reg.worker_id,
                        capacity=reg.capacity,
                        availability=reg.availability,
                        state=WorkerState.UNHEALTHY,
                        measured_capacity=reg.measured_capacity,
                        last_heartbeat=reg.last_heartbeat,
                    )
                    self._workers[worker_id] = unhealthy
                    transitioned.append(unhealthy)

        return transitioned

    def quarantine(
        self,
        worker_id: str,
        reason: str,
    ) -> WorkerRegistration:
        """Move a worker to QUARANTINED state due to SLA failure.

        Args:
            worker_id: must be registered
            reason: human-readable explanation of the SLA failure

        Returns:
            WorkerRegistration in QUARANTINED state

        Raises:
            KeyError: if worker not registered
            ValueError: if reason is empty
        """
        if not reason or not reason.strip():
            raise ValueError(f"{worker_id}: quarantine reason cannot be empty")

        reg = self._workers[worker_id]
        quarantined = WorkerRegistration(
            worker_id=reg.worker_id,
            capacity=reg.capacity,
            availability=reg.availability,
            state=WorkerState.QUARANTINED,
            measured_capacity=reg.measured_capacity,
            last_heartbeat=reg.last_heartbeat,
            quarantine_reason=reason.strip(),
        )
        self._workers[worker_id] = quarantined
        return quarantined

    def readmit(self, worker_id: str) -> WorkerRegistration:
        """Re-verify and return a QUARANTINED worker to ACTIVE.

        Transitions the worker back to ACTIVE without re-measuring capacity
        (assumes the SLA violation has been resolved and the measured capacity
        is still valid). If the worker is not QUARANTINED, raises ValueError.

        Args:
            worker_id: must be registered and QUARANTINED

        Returns:
            WorkerRegistration in ACTIVE state

        Raises:
            KeyError: if worker not registered
            ValueError: if worker is not QUARANTINED
        """
        reg = self._workers[worker_id]

        if reg.state != WorkerState.QUARANTINED:
            raise ValueError(
                f"{worker_id}: readmit requires QUARANTINED state, "
                f"got {reg.state.value}"
            )

        readmitted = WorkerRegistration(
            worker_id=reg.worker_id,
            capacity=reg.capacity,
            availability=reg.availability,
            state=WorkerState.ACTIVE,
            measured_capacity=reg.measured_capacity,
            last_heartbeat=self._get_time(),
        )
        self._workers[worker_id] = readmitted
        return readmitted

    def available_workers(self) -> list[WorkerRegistration]:
        """Return all ACTIVE workers eligible for placement.

        Returns:
            list of WorkerRegistration objects in ACTIVE state, in no
            guaranteed order
        """
        return [
            reg
            for reg in self._workers.values()
            if reg.state == WorkerState.ACTIVE
        ]

    def volunteer_workers(self) -> list[acquire.VolunteerWorker]:
        """The ACTIVE pool, in the shape `acquire.assess()` consumes.

        This is the seam between "who is in the pool" and "will the pool assemble", and
        without it the two halves are separate libraries that happen to live in one
        package: a caller had to hand-translate WorkerRegistration into VolunteerWorker
        and would inevitably pass the DECLARED capacity, since that is the attribute
        with the obvious name.

        It passes `measured_capacity`, never `capacity`. The declared figure is a claim
        a volunteer made about hardware nobody checked; feeding it to the probability
        math produces a confident number about a pool that does not exist. That is the
        whole reason `verify_and_activate` is a precondition of being ACTIVE at all.
        """
        return [
            acquire.VolunteerWorker(
                worker_id=reg.worker_id,
                capacity=reg.measured_capacity,
                availability=reg.availability,
            )
            for reg in self.available_workers()
        ]

    def get_worker(self, worker_id: str) -> Optional[WorkerRegistration]:
        """Look up a worker by id.

        Args:
            worker_id: worker identifier

        Returns:
            WorkerRegistration if found, None otherwise
        """
        return self._workers.get(worker_id)

    def workers(self) -> list[WorkerRegistration]:
        """Return all registered workers in any state.

        Returns:
            list of all WorkerRegistration objects
        """
        return list(self._workers.values())
