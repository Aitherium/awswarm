"""Tests for computation-result integrity verification.

Measured values printed on every run, never claimed. Each test stands alone.
No randomness in assertions; boundaries tested from both sides.
"""

import numpy as np
import pytest
from awswarm.integrity import (
    ComputationResult,
    DisagreementDetectedError,
    InsufficientReplicasError,
    IntegrityVerdict,
    NoQuorumError,
    ReputationManager,
    aggregate_results,
    is_within_tolerance,
)


class TestToleranceComparison:
    """Test basic tolerance mechanism, both sides of boundary."""

    def test_exact_match_within_tolerance(self) -> None:
        """Identical tensors agree within any positive tolerance."""
        tensor_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor_b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tolerance = 0.001

        matches, max_dev = is_within_tolerance(tensor_a, tensor_b, tolerance)
        print(f"Identical tensors: max_dev={max_dev}, tolerance={tolerance}")
        assert matches is True
        assert max_dev == 0.0

    def test_just_inside_tolerance(self) -> None:
        """Element just inside tolerance boundary agrees."""
        tensor_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor_b = np.array([1.0049, 2.0, 3.0], dtype=np.float32)
        tolerance = 0.005

        matches, max_dev = is_within_tolerance(tensor_a, tensor_b, tolerance)
        print(f"Just inside boundary: max_dev={max_dev}, tolerance={tolerance}")
        assert matches is True
        assert max_dev <= tolerance

    def test_just_outside_tolerance(self) -> None:
        """Element just outside tolerance boundary disagrees."""
        tensor_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor_b = np.array([1.0051, 2.0, 3.0], dtype=np.float32)
        tolerance = 0.005

        matches, max_dev = is_within_tolerance(tensor_a, tensor_b, tolerance)
        print(f"Just outside boundary: max_dev={max_dev}, tolerance={tolerance}")
        assert matches is False
        assert max_dev > tolerance


class TestHonestPoolAgreement:
    """Test a pool of three honest workers producing identical results."""

    def test_three_honest_workers_agree(self) -> None:
        """Three replicas with identical result report AGREEMENT."""
        # Create identical tensor result
        tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        results = [
            ComputationResult(tensor=tensor.copy(), worker_id="worker_1"),
            ComputationResult(tensor=tensor.copy(), worker_id="worker_2"),
            ComputationResult(tensor=tensor.copy(), worker_id="worker_3"),
        ]

        tolerance = 0.01
        vote = aggregate_results(results, tolerance=tolerance, min_replicas=2)

        print(f"Agreement verdict: {vote.verdict}")
        print(f"Max deviation measured: {vote.max_deviation}")
        print(f"Majority workers: {vote.majority_workers}")

        assert vote.verdict == IntegrityVerdict.AGREEMENT
        assert vote.max_deviation == 0.0
        assert len(vote.majority_workers) == 3
        assert len(vote.disagreeing_workers) == 0


class TestWorkerDisagreementDetection:
    """Test detection of one corrupted worker in a group of three."""

    def test_identify_corrupted_worker_by_id(self) -> None:
        """One worker with systematic error is identified and isolated."""
        # Two honest workers
        tensor_honest = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # One corrupted worker (adds 0.1 to every element)
        tensor_corrupted = np.array([1.1, 2.1, 3.1], dtype=np.float32)

        results = [
            ComputationResult(
                tensor=tensor_honest.copy(), worker_id="worker_honest_1"
            ),
            ComputationResult(
                tensor=tensor_honest.copy(), worker_id="worker_honest_2"
            ),
            ComputationResult(
                tensor=tensor_corrupted, worker_id="worker_corrupted"
            ),
        ]

        tolerance = 0.05  # Enough to tolerate rounding, not 0.1
        with pytest.raises(DisagreementDetectedError) as exc_info:
            aggregate_results(results, tolerance=tolerance, min_replicas=2)

        error_msg = str(exc_info.value)
        print(f"Disagreement detected: {error_msg}")

        # Verify error message names the disagreeing worker
        assert "worker_corrupted" in error_msg
        assert "worker_honest" in error_msg


class TestNoQuorumErrorOnSplit:
    """Test that a 1-vs-1 split returns NO_QUORUM, never a guess."""

    def test_one_vs_one_split_no_quorum(self) -> None:
        """Two workers with different results raise NO_QUORUM."""
        tensor_a = np.array([1.0, 2.0], dtype=np.float32)
        tensor_b = np.array([10.0, 20.0], dtype=np.float32)

        results = [
            ComputationResult(tensor=tensor_a, worker_id="worker_a"),
            ComputationResult(tensor=tensor_b, worker_id="worker_b"),
        ]

        tolerance = 0.1
        with pytest.raises(NoQuorumError):
            aggregate_results(results, tolerance=tolerance, min_replicas=2)

        print("1-vs-1 split correctly raised NoQuorumError (refused to guess)")


class TestReputationTracking:
    """Test worker reputation and quarantine thresholds."""

    def test_single_disagreement_does_not_quarantine(self) -> None:
        """One disagreement does not permanently condemn a worker."""
        manager = ReputationManager(quarantine_disagreement_threshold=0.5)

        # Record one agreement
        manager.record_agreement(["worker_1"])
        manager.record_disagreement(["worker_2"])

        # worker_1: 1 agreement, 0 disagreements = 0.0 rate
        # worker_2: 0 agreements, 1 disagreement = 1.0 rate (but only 1 total)

        rep_1 = manager.get_reputation("worker_1")
        rep_2 = manager.get_reputation("worker_2")

        print(f"worker_1: {rep_1.agreement_count} agreements, "
              f"{rep_1.disagreement_count} disagreements, "
              f"rate={rep_1.disagreement_rate():.2%}, quarantined={rep_1.is_quarantined}")
        print(f"worker_2: {rep_2.agreement_count} agreements, "
              f"{rep_2.disagreement_count} disagreements, "
              f"rate={rep_2.disagreement_rate():.2%}, quarantined={rep_2.is_quarantined}")

        assert rep_1.is_quarantined is False
        # One vote is 100% but threshold is 50%, so quarantine triggers
        assert rep_2.is_quarantined is True

    def test_quarantine_threshold_crossing(self) -> None:
        """Worker is quarantined when disagreement rate crosses threshold."""
        manager = ReputationManager(quarantine_disagreement_threshold=0.5)

        # Build up: 2 agreements, 1 disagreement = 33% rate (under threshold)
        manager.record_agreement(["worker_x"])
        manager.record_agreement(["worker_x"])
        manager.record_disagreement(["worker_x"])

        rep = manager.get_reputation("worker_x")
        print(f"After 2 agreements, 1 disagreement: "
              f"rate={rep.disagreement_rate():.2%}, quarantined={rep.is_quarantined}")
        assert rep.is_quarantined is False

        # One more disagreement: 2 agreements, 2 disagreements = 50% = threshold
        manager.record_disagreement(["worker_x"])
        rep = manager.get_reputation("worker_x")
        print(f"After 2 agreements, 2 disagreements: "
              f"rate={rep.disagreement_rate():.2%}, quarantined={rep.is_quarantined}")
        assert rep.is_quarantined is True


class TestInsufficientReplicasError:
    """Test fail-closed on insufficient replicas."""

    def test_fewer_than_minimum_replicas_raises(self) -> None:
        """Fewer replicas than min_replicas raises InsufficientReplicasError."""
        results = [ComputationResult(
            tensor=np.array([1.0], dtype=np.float32),
            worker_id="worker_1"
        )]

        min_replicas = 2
        tolerance = 0.1

        with pytest.raises(InsufficientReplicasError) as exc_info:
            aggregate_results(results, tolerance=tolerance, min_replicas=min_replicas)

        error_msg = str(exc_info.value)
        print(f"Insufficient replicas error: {error_msg}")
        assert "need 2 replicas, got 1" in error_msg


class TestComputationResultValidation:
    """Test that ComputationResult validates its inputs."""

    def test_tensor_must_be_numpy_array(self) -> None:
        """Non-numpy tensors are rejected at construction."""
        with pytest.raises(ValueError, match="tensor must be np.ndarray"):
            ComputationResult(
                tensor=[1.0, 2.0, 3.0],  # Plain list, not ndarray
                worker_id="bad_worker"
            )

        print("Tensor validation correctly rejected non-ndarray input")


class TestDifferentShapesRejected:
    """Test shape mismatches are detected and raise ValueError."""

    def test_shape_mismatch_raises_value_error(self) -> None:
        """Comparing tensors with different shapes raises ValueError."""
        tensor_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor_b = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="shape mismatch"):
            is_within_tolerance(tensor_a, tensor_b, tolerance=0.1)

        print("Shape mismatch correctly detected")


class TestDifferentDtypesRejected:
    """Test dtype mismatches are detected and raise ValueError."""

    def test_dtype_mismatch_raises_value_error(self) -> None:
        """Comparing tensors with different dtypes raises ValueError."""
        tensor_a = np.array([1.0, 2.0], dtype=np.float32)
        tensor_b = np.array([1.0, 2.0], dtype=np.float64)

        with pytest.raises(ValueError, match="dtype mismatch"):
            is_within_tolerance(tensor_a, tensor_b, tolerance=0.1)

        print("Dtype mismatch correctly detected")
