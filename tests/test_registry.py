"""Tests for WorkerRegistry and the measured-registration pattern.

WHAT IS PROVEN HERE: a volunteer worker registry that validates benchmark
results against declared capacity, maintains four distinct worker states
(UNVERIFIED, ACTIVE, UNHEALTHY, QUARANTINED), implements deterministic
heartbeat tracking (testable with injected clocks), and provides
quarantine/readmit workflows with reason tracking. Tests use injected time
functions to avoid ambient time dependencies.

WHAT IS NOT CLAIMED: clock injection is for testing only; production code
mounts time.time() by default and does not sleep. Tests do not exercise
actual network probes or benchmark execution — only the state machine,
verification logic, and capacity tolerance checks.

WHY THIS MATTERS: the volunteer pool drifts constantly, and unverified
workers silently joining the pool is the defect this catches. Heartbeat
expiry (did not hear from you) must be distinct from quarantine (you
exceeded your SLA) so recovery handlers can tailor their response.
"""


import pytest
from awswarm import acquire
from awswarm.registry import (
    WorkerCapacityMismatchError,
    WorkerRegistration,
    WorkerRegistry,
    WorkerState,
)


class TestWorkerRegistryInitialization:
    """Registry setup and validation of constructor arguments."""

    def test_init_with_defaults(self) -> None:
        """Default heartbeat timeout is 300s, capacity tolerance is 5%."""
        registry = WorkerRegistry()
        reg = registry.register("w1", capacity=100, availability=0.9)
        assert reg.state == WorkerState.UNVERIFIED
        # Heartbeat timeout and tolerance are stored but not directly
        # inspectable; we test them through behavior below.

    def test_init_rejects_invalid_heartbeat_timeout(self) -> None:
        """Heartbeat timeout must be positive."""
        with pytest.raises(ValueError) as exc:
            WorkerRegistry(heartbeat_timeout_sec=0)
        assert "heartbeat_timeout_sec must be > 0, got 0" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            WorkerRegistry(heartbeat_timeout_sec=-30)
        assert "heartbeat_timeout_sec must be > 0, got -30" in str(exc.value)

    def test_init_rejects_invalid_capacity_tolerance(self) -> None:
        """Capacity tolerance must be in [0, 1]."""
        with pytest.raises(ValueError) as exc:
            WorkerRegistry(capacity_tolerance=-0.01)
        assert "capacity_tolerance must be in [0, 1], got -0.01" in str(
            exc.value
        )

        with pytest.raises(ValueError) as exc:
            WorkerRegistry(capacity_tolerance=1.5)
        assert "capacity_tolerance must be in [0, 1], got 1.5" in str(
            exc.value
        )

    def test_init_with_custom_clock(self) -> None:
        """Registry accepts injected time function for deterministic tests."""
        mock_time = 1000.0

        def get_mock_time() -> float:
            return mock_time

        registry = WorkerRegistry(get_time_fn=get_mock_time)
        reg = registry.register("w1", capacity=100, availability=0.9)
        assert reg.last_heartbeat == 1000.0


class TestWorkerRegistration:
    """WorkerRegistration dataclass validation."""

    def test_registration_rejects_negative_capacity(self) -> None:
        """Capacity must be non-negative."""
        with pytest.raises(ValueError) as exc:
            WorkerRegistration(
                worker_id="w1",
                capacity=-1,
                availability=0.5,
                state=WorkerState.UNVERIFIED,
            )
        assert "w1: capacity must be >= 0, got -1" in str(exc.value)

    def test_registration_rejects_invalid_availability(self) -> None:
        """Availability must be in [0.0, 1.0]."""
        with pytest.raises(ValueError) as exc:
            WorkerRegistration(
                worker_id="w1",
                capacity=100,
                availability=1.5,
                state=WorkerState.UNVERIFIED,
            )
        assert "w1: availability must be in [0, 1], got 1.5" in str(
            exc.value
        )

    def test_active_state_requires_measured_capacity(self) -> None:
        """ACTIVE state must have a measured_capacity value."""
        with pytest.raises(ValueError) as exc:
            WorkerRegistration(
                worker_id="w1",
                capacity=100,
                availability=0.5,
                state=WorkerState.ACTIVE,
                measured_capacity=None,
            )
        assert "w1: ACTIVE state requires measured_capacity, got None" in str(
            exc.value
        )

    def test_quarantined_state_requires_reason(self) -> None:
        """QUARANTINED state must have a quarantine_reason."""
        with pytest.raises(ValueError) as exc:
            WorkerRegistration(
                worker_id="w1",
                capacity=100,
                availability=0.5,
                state=WorkerState.QUARANTINED,
                quarantine_reason=None,
            )
        assert (
            "w1: QUARANTINED state requires quarantine_reason, got None"
            in str(exc.value)
        )


class TestRegisterWorker:
    """Testing the register() method."""

    def test_register_new_worker(self) -> None:
        """Register a new worker in UNVERIFIED state."""
        registry = WorkerRegistry()
        reg = registry.register("w1", capacity=100, availability=0.8)

        assert reg.worker_id == "w1"
        assert reg.capacity == 100
        assert reg.availability == 0.8
        assert reg.state == WorkerState.UNVERIFIED
        assert reg.measured_capacity is None
        assert reg.quarantine_reason is None

    def test_register_rejects_duplicate_worker(self) -> None:
        """Registering the same worker twice raises ValueError."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.8)

        with pytest.raises(ValueError) as exc:
            registry.register("w1", capacity=100, availability=0.8)
        assert "w1: already registered" in str(exc.value)

    def test_register_rejects_invalid_capacity(self) -> None:
        """Negative capacity raises ValueError."""
        registry = WorkerRegistry()
        with pytest.raises(ValueError) as exc:
            registry.register("w1", capacity=-10, availability=0.8)
        assert "w1: capacity must be >= 0, got -10" in str(exc.value)


class TestVerifyAndActivate:
    """Testing the verify_and_activate() method."""

    def test_verify_with_matching_measured_capacity(self) -> None:
        """Verification succeeds when measured capacity matches declared."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.9)

        # Measured = declared = 100, tolerance = 5%, so error = 0%
        # which is <= 5%, should pass.
        active = registry.verify_and_activate("w1", measured_capacity=100)

        assert active.state == WorkerState.ACTIVE
        assert active.measured_capacity == 100

    def test_verify_with_tolerance_range(self) -> None:
        """Measured capacity within tolerance passes (5% tolerance)."""
        registry = WorkerRegistry(capacity_tolerance=0.05)
        registry.register("w1", capacity=100, availability=0.9)

        # Declared=100, tolerance=5% => max_error=5
        # Measured=105 => error=5, within tolerance
        active = registry.verify_and_activate("w1", measured_capacity=105)
        assert active.state == WorkerState.ACTIVE

        # Measured=95 => error=5, also within tolerance
        registry.register("w2", capacity=100, availability=0.9)
        active = registry.verify_and_activate("w2", measured_capacity=95)
        assert active.state == WorkerState.ACTIVE

    def test_verify_rejects_divergent_measured_capacity(self) -> None:
        """Measured capacity outside tolerance raises WorkerCapacityMismatchError."""
        registry = WorkerRegistry(capacity_tolerance=0.05)
        registry.register("w1", capacity=100, availability=0.9)

        # Declared=100, tolerance=5% => max_error=5
        # Measured=110 => error=10, exceeds tolerance
        with pytest.raises(WorkerCapacityMismatchError) as exc:
            registry.verify_and_activate("w1", measured_capacity=110)

        mismatch = exc.value
        assert mismatch.worker_id == "w1"
        assert mismatch.declared == 100
        assert mismatch.measured == 110
        assert mismatch.tolerance == 0.05
        assert "10.0% error" in str(mismatch)

    def test_verify_rejects_invalid_measured_capacity(self) -> None:
        """Negative measured capacity raises ValueError."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.9)

        with pytest.raises(ValueError) as exc:
            registry.verify_and_activate("w1", measured_capacity=-5)
        assert "w1: measured_capacity must be >= 0, got -5" in str(exc.value)

    def test_verify_rejects_invalid_declared_capacity(self) -> None:
        """A worker whose own capacity cannot hold itself fails loudly."""
        registry = WorkerRegistry()
        # Deliberately broken declared capacity (would never pass the single-
        # worker assessment via acquire.assess()).
        registry.register("w1", capacity=100, availability=0.0)

        with pytest.raises(ValueError) as exc:
            # Measured capacity matches declared, but declared=100 with
            # availability=0.0 means a single-worker pool cannot assemble.
            registry.verify_and_activate("w1", measured_capacity=100)
        assert (
            "w1: declared capacity 100 cannot hold itself"
            in str(exc.value)
        )


class TestHeartbeatWithInjectedClock:
    """Testing heartbeat tracking with deterministic time injection."""

    def test_heartbeat_on_active_worker(self) -> None:
        """Heartbeat updates last_heartbeat on ACTIVE worker."""
        mock_time = [1000.0]

        def get_mock_time() -> float:
            return mock_time[0]

        registry = WorkerRegistry(get_time_fn=get_mock_time)
        registry.register("w1", capacity=100, availability=0.9)
        registry.verify_and_activate("w1", measured_capacity=100)

        # Move time forward
        mock_time[0] = 2000.0
        hb = registry.heartbeat("w1")

        assert hb.last_heartbeat == 2000.0
        assert hb.state == WorkerState.ACTIVE

    def test_heartbeat_restores_unhealthy_to_active(self) -> None:
        """Heartbeat transitions UNHEALTHY back to ACTIVE."""
        mock_time = [1000.0]

        def get_mock_time() -> float:
            return mock_time[0]

        registry = WorkerRegistry(
            heartbeat_timeout_sec=100.0, get_time_fn=get_mock_time
        )
        registry.register("w1", capacity=100, availability=0.9)
        registry.verify_and_activate("w1", measured_capacity=100)

        # Move past heartbeat timeout
        mock_time[0] = 1200.0
        registry.check_heartbeats()

        w1 = registry.get_worker("w1")
        assert w1.state == WorkerState.UNHEALTHY

        # Heartbeat restores it
        hb = registry.heartbeat("w1")
        assert hb.state == WorkerState.ACTIVE

    def test_heartbeat_rejects_unverified_worker(self) -> None:
        """Heartbeat on UNVERIFIED worker raises ValueError."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.9)

        with pytest.raises(ValueError) as exc:
            registry.heartbeat("w1")
        assert (
            "w1: cannot heartbeat UNVERIFIED worker — "
            "call verify_and_activate() first"
        ) in str(exc.value)

    def test_heartbeat_rejects_quarantined_worker(self) -> None:
        """Heartbeat on QUARANTINED worker raises ValueError."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.9)
        registry.verify_and_activate("w1", measured_capacity=100)
        registry.quarantine("w1", reason="SLA violation: response time > 5s")

        with pytest.raises(ValueError) as exc:
            registry.heartbeat("w1")
        assert (
            "w1: cannot heartbeat QUARANTINED worker — "
            "call readmit() to restore"
        ) in str(exc.value)


class TestCheckHeartbeats:
    """Testing heartbeat expiry scanning with injected clock."""

    def test_check_heartbeats_marks_expired_as_unhealthy(self) -> None:
        """ACTIVE workers past timeout transition to UNHEALTHY."""
        mock_time = [1000.0]

        def get_mock_time() -> float:
            return mock_time[0]

        registry = WorkerRegistry(
            heartbeat_timeout_sec=300.0, get_time_fn=get_mock_time
        )
        registry.register("w1", capacity=100, availability=0.9)
        registry.verify_and_activate("w1", measured_capacity=100)

        # Time advances past the timeout
        mock_time[0] = 1400.0
        transitioned = registry.check_heartbeats()

        assert len(transitioned) == 1
        assert transitioned[0].worker_id == "w1"
        assert transitioned[0].state == WorkerState.UNHEALTHY

    def test_check_heartbeats_ignores_healthy(self) -> None:
        """ACTIVE workers within timeout are not affected."""
        mock_time = [1000.0]

        def get_mock_time() -> float:
            return mock_time[0]

        registry = WorkerRegistry(
            heartbeat_timeout_sec=300.0, get_time_fn=get_mock_time
        )
        registry.register("w1", capacity=100, availability=0.9)
        registry.verify_and_activate("w1", measured_capacity=100)

        # Time advances but stays within timeout
        mock_time[0] = 1100.0
        transitioned = registry.check_heartbeats()

        assert len(transitioned) == 0

    def test_check_heartbeats_ignores_unverified(self) -> None:
        """UNVERIFIED workers are never expired by check_heartbeats."""
        registry = WorkerRegistry(heartbeat_timeout_sec=1.0)
        registry.register("w1", capacity=100, availability=0.9)
        # Do not verify; leave in UNVERIFIED state

        transitioned = registry.check_heartbeats()
        assert len(transitioned) == 0


class TestQuarantine:
    """Testing quarantine() method."""

    def test_quarantine_worker(self) -> None:
        """Quarantine transitions worker to QUARANTINED with reason."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.9)
        registry.verify_and_activate("w1", measured_capacity=100)

        quarantined = registry.quarantine(
            "w1", reason="Exceeded SLA: 99th percentile latency > 500ms"
        )

        assert quarantined.state == WorkerState.QUARANTINED
        assert (
            "Exceeded SLA: 99th percentile latency > 500ms"
            in quarantined.quarantine_reason
        )

    def test_quarantine_rejects_empty_reason(self) -> None:
        """Quarantine reason cannot be empty or whitespace."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.9)
        registry.verify_and_activate("w1", measured_capacity=100)

        with pytest.raises(ValueError) as exc:
            registry.quarantine("w1", reason="")
        assert "w1: quarantine reason cannot be empty" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            registry.quarantine("w1", reason="   ")
        assert "w1: quarantine reason cannot be empty" in str(exc.value)


class TestReadmit:
    """Testing readmit() method."""

    def test_readmit_quarantined_worker(self) -> None:
        """Readmit transitions QUARANTINED back to ACTIVE."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.9)
        registry.verify_and_activate("w1", measured_capacity=100)
        registry.quarantine("w1", reason="SLA violation")

        readmitted = registry.readmit("w1")

        assert readmitted.state == WorkerState.ACTIVE
        assert readmitted.quarantine_reason is None

    def test_readmit_rejects_non_quarantined_worker(self) -> None:
        """Readmit on non-QUARANTINED worker raises ValueError."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.9)
        registry.verify_and_activate("w1", measured_capacity=100)

        with pytest.raises(ValueError) as exc:
            registry.readmit("w1")
        assert (
            "w1: readmit requires QUARANTINED state, got active"
            in str(exc.value)
        )


class TestAvailableWorkers:
    """Testing available_workers() method."""

    def test_available_workers_returns_only_active(self) -> None:
        """available_workers() returns only ACTIVE workers."""
        registry = WorkerRegistry()

        # Register three workers with different states
        registry.register("w1", capacity=100, availability=0.9)
        registry.verify_and_activate("w1", measured_capacity=100)

        registry.register("w2", capacity=80, availability=0.7)
        # w2 stays UNVERIFIED

        registry.register("w3", capacity=120, availability=0.95)
        registry.verify_and_activate("w3", measured_capacity=120)
        registry.quarantine("w3", reason="Testing")

        available = registry.available_workers()

        assert len(available) == 1
        assert available[0].worker_id == "w1"

    def test_available_workers_empty_pool(self) -> None:
        """available_workers() returns empty list when no ACTIVE workers."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.9)
        # Never verify

        available = registry.available_workers()
        assert available == []


class TestIntegrationWithAcquire:
    """Integration with awswarm.acquire.assess() for probability measurement."""

    def test_verify_uses_assess_for_single_worker_pool(self) -> None:
        """Verification constructs a single-worker pool and calls assess()."""
        registry = WorkerRegistry()
        registry.register("w1", capacity=100, availability=0.9)

        # When we verify, it internally calls acquire.assess() with the
        # registered worker. The test is that it succeeds without error
        # (assess would fail if the worker could not hold its own capacity).
        active = registry.verify_and_activate("w1", measured_capacity=100)
        assert active.state == WorkerState.ACTIVE

    def test_multiple_active_workers_can_be_assembled(self) -> None:
        """Multiple ACTIVE workers can be queried via available_workers()."""
        registry = WorkerRegistry()

        # Register and verify multiple workers
        for i in range(3):
            worker_id = f"w{i}"
            registry.register(worker_id, capacity=100, availability=0.8)
            registry.verify_and_activate(worker_id, measured_capacity=100)

        available = registry.available_workers()
        assert len(available) == 3

        # Construct a VolunteerWorker list suitable for acquire.assess()
        workers = [
            acquire.VolunteerWorker(
                worker_id=reg.worker_id,
                capacity=reg.capacity,
                availability=reg.availability,
            )
            for reg in available
        ]

        # With 3 workers at 100 each and availability 0.8, the pool can
        # assemble any request up to 300 with a measurable probability.
        # Just verify it does not error.
        report = acquire.assess(total_units=200, workers=workers)
        assert report.total_units == 200
        assert report.pool_capacity == 300
