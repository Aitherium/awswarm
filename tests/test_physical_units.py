"""Granular unit coverage for awswarm.physical.

Written by the module's build agent, which wrote it to `test/` (singular) rather than
`tests/` and then died before running it. Two consequences, both found 2026-09-03:
pytest's module naming COLLIDED with tests/test_physical.py and errored the whole
package-root collection while SHADOWING the behavioural file, and every "public
address" case here used RFC 5737 / RFC 3849 documentation ranges, which Python's
ipaddress reports as PRIVATE -- so 8 cases failed the moment they were first executed.

Kept alongside the behavioural file rather than merged: this covers per-input
validation, that one covers both-direction policy. Renamed for a unique basename,
because two test modules sharing one basename with no __init__.py is what caused the
collision.
"""

import pytest
from awswarm.physical import ExecutionLabel, Worker, _is_externally_reachable, classify


class TestWorkerConstruction:
    """Verify Worker dataclass validates its inputs."""

    def test_worker_valid(self) -> None:
        """Worker accepts valid hostname and address."""
        w = Worker(hostname="host1", address="8.8.8.1")
        assert w.hostname == "host1"
        assert w.address == "8.8.8.1"

    def test_worker_empty_hostname(self) -> None:
        """Worker rejects empty hostname."""
        with pytest.raises(ValueError, match="hostname must not be empty"):
            Worker(hostname="", address="8.8.8.1")

    def test_worker_whitespace_hostname(self) -> None:
        """Worker rejects whitespace-only hostname."""
        with pytest.raises(ValueError, match="hostname must not be empty"):
            Worker(hostname="   ", address="8.8.8.1")

    def test_worker_empty_address(self) -> None:
        """Worker rejects empty address."""
        with pytest.raises(ValueError, match="address must not be empty"):
            Worker(hostname="host1", address="")

    def test_worker_whitespace_address(self) -> None:
        """Worker rejects whitespace-only address."""
        with pytest.raises(ValueError, match="address must not be empty"):
            Worker(hostname="host1", address="   ")

    def test_worker_invalid_ipv4(self) -> None:
        """Worker rejects invalid IPv4 address."""
        with pytest.raises(ValueError, match="not a valid IP address"):
            Worker(hostname="host1", address="256.256.256.256")

    def test_worker_invalid_ipv6(self) -> None:
        """Worker rejects invalid IPv6 address."""
        with pytest.raises(ValueError, match="not a valid IP address"):
            Worker(hostname="host1", address="gggg::1")


class TestExternallyReachable:
    """Verify address classification."""

    def test_ipv4_loopback_rejected(self) -> None:
        """Loopback IPv4 (127.0.0.0/8) is not externally reachable."""
        assert not _is_externally_reachable("127.0.0.1")
        assert not _is_externally_reachable("127.255.255.255")

    def test_ipv6_loopback_rejected(self) -> None:
        """Loopback IPv6 (::1) is not externally reachable."""
        assert not _is_externally_reachable("::1")

    def test_rfc1918_10_rejected(self) -> None:
        """RFC1918 10.0.0.0/8 is not externally reachable."""
        assert not _is_externally_reachable("10.0.0.0")
        assert not _is_externally_reachable("10.255.255.255")

    def test_rfc1918_172_rejected(self) -> None:
        """RFC1918 172.16.0.0/12 is not externally reachable."""
        assert not _is_externally_reachable("172.16.0.0")
        assert not _is_externally_reachable("172.31.255.255")

    def test_rfc1918_192_rejected(self) -> None:
        """RFC1918 192.168.0.0/16 is not externally reachable."""
        assert not _is_externally_reachable("192.168.0.0")
        assert not _is_externally_reachable("192.168.255.255")

    def test_ipv4_link_local_rejected(self) -> None:
        """Link-local IPv4 (169.254.0.0/16) is not externally reachable."""
        assert not _is_externally_reachable("169.254.0.0")
        assert not _is_externally_reachable("169.254.255.255")

    def test_ipv6_link_local_rejected(self) -> None:
        """Link-local IPv6 (fe80::/10) is not externally reachable."""
        assert not _is_externally_reachable("fe80::1")
        assert not _is_externally_reachable("fe80::ffff:ffff:ffff:ffff")

    def test_public_ipv4_accepted(self) -> None:
        """Public IPv4 addresses are externally reachable."""
        assert _is_externally_reachable("8.8.8.1")
        assert _is_externally_reachable("8.8.8.8")
        assert _is_externally_reachable("1.1.1.1")

    def test_public_ipv6_accepted(self) -> None:
        """Public IPv6 addresses are externally reachable."""
        assert _is_externally_reachable("2606:4700:4700::1")
        assert _is_externally_reachable("2001:4860:4860::8888")


class TestClassifyValidation:
    """Verify classify() enforces preconditions."""

    def test_empty_workers_rejected(self) -> None:
        """Cannot classify with zero workers."""
        with pytest.raises(ValueError, match="cannot classify distribution"):
            classify([], "8.8.8.1")

    def test_invalid_controller_address(self) -> None:
        """Invalid controller address is rejected."""
        workers = [Worker(hostname="h1", address="8.8.8.1")]
        with pytest.raises(ValueError, match="not a valid IP address"):
            classify(workers, "not-an-ip")

    def test_duplicate_hostnames_rejected(self) -> None:
        """Two workers cannot share a hostname."""
        workers = [
            Worker(hostname="host1", address="8.8.8.1"),
            Worker(hostname="host1", address="8.8.8.2"),
        ]
        with pytest.raises(ValueError, match="cannot share a hostname"):
            classify(workers, "8.8.8.10")


class TestClassifySimulated:
    """SIMULATED label when explicitly marked."""

    def test_simulated_true_ignored_all_evidence(self) -> None:
        """Simulated flag overrides all other evidence."""
        # Even with all external addresses and distinct hostnames,
        # simulated=True returns SIMULATED.
        workers = [
            Worker(hostname="h1", address="8.8.8.1"),
            Worker(hostname="h2", address="8.8.8.2"),
        ]
        label = classify(workers, "8.8.8.10", simulated=True)
        assert label == ExecutionLabel.SIMULATED

    def test_simulated_false_normal_classification(self) -> None:
        """Simulated=False (default) follows normal classification."""
        workers = [Worker(hostname="h1", address="8.8.8.1")]
        # Single worker on external address, controller is different.
        label = classify(workers, "8.8.8.10", simulated=False)
        assert label == ExecutionLabel.PHYSICAL


class TestClassifyLoopback:
    """LOOPBACK label for co-located or private deployments."""

    def test_private_address_returns_loopback(self) -> None:
        """Worker with private address returns LOOPBACK."""
        workers = [Worker(hostname="h1", address="192.168.1.1")]
        label = classify(workers, "8.8.8.10")
        assert label == ExecutionLabel.LOOPBACK

    def test_ipv4_loopback_address_returns_loopback(self) -> None:
        """Worker with IPv4 loopback returns LOOPBACK."""
        workers = [Worker(hostname="h1", address="127.0.0.1")]
        label = classify(workers, "8.8.8.10")
        assert label == ExecutionLabel.LOOPBACK

    def test_ipv6_loopback_address_returns_loopback(self) -> None:
        """Worker with IPv6 loopback returns LOOPBACK."""
        workers = [Worker(hostname="h1", address="::1")]
        label = classify(workers, "8.8.8.10")
        assert label == ExecutionLabel.LOOPBACK

    def test_link_local_returns_loopback(self) -> None:
        """Worker with link-local address returns LOOPBACK."""
        workers = [Worker(hostname="h1", address="169.254.1.1")]
        label = classify(workers, "8.8.8.10")
        assert label == ExecutionLabel.LOOPBACK

    def test_controller_same_as_worker_returns_loopback(self) -> None:
        """Worker on same address as controller returns LOOPBACK."""
        workers = [Worker(hostname="h1", address="8.8.8.1")]
        label = classify(workers, "8.8.8.1")
        assert label == ExecutionLabel.LOOPBACK

    def test_multiple_workers_one_private_returns_loopback(self) -> None:
        """Any worker on private address returns LOOPBACK."""
        workers = [
            Worker(hostname="h1", address="8.8.8.1"),
            Worker(hostname="h2", address="192.168.1.1"),  # private
        ]
        label = classify(workers, "8.8.8.10")
        assert label == ExecutionLabel.LOOPBACK


class TestClassifyPhysical:
    """PHYSICAL label for genuinely distributed deployments."""

    def test_single_external_worker_distinct_from_controller(self) -> None:
        """Single worker on external address distinct from controller is PHYSICAL."""
        workers = [Worker(hostname="h1", address="8.8.8.1")]
        label = classify(workers, "8.8.8.10")
        assert label == ExecutionLabel.PHYSICAL

    def test_multiple_external_workers_distinct_hostnames_distinct_from_controller(
        self,
    ) -> None:
        """Multiple external workers on distinct hostnames and distinct from controller
        is PHYSICAL."""
        workers = [
            Worker(hostname="h1", address="8.8.8.1"),
            Worker(hostname="h2", address="8.8.8.2"),
            Worker(hostname="h3", address="8.8.8.3"),
        ]
        label = classify(workers, "8.8.8.10")
        assert label == ExecutionLabel.PHYSICAL

    def test_ipv6_external_workers_are_physical(self) -> None:
        """External IPv6 addresses also count as PHYSICAL."""
        workers = [
            Worker(hostname="h1", address="2606:4700:4700::1"),
            Worker(hostname="h2", address="2606:4700:4700::2"),
        ]
        label = classify(workers, "2606:4700:4700::10")
        assert label == ExecutionLabel.PHYSICAL

    def test_mixed_ipv4_ipv6_external_workers_are_physical(self) -> None:
        """Mixing IPv4 and IPv6 public addresses is PHYSICAL."""
        workers = [
            Worker(hostname="h1", address="8.8.8.1"),
            Worker(hostname="h2", address="2606:4700:4700::1"),
        ]
        label = classify(workers, "8.8.8.10")
        assert label == ExecutionLabel.PHYSICAL

    def test_large_cluster_external_addresses_is_physical(self) -> None:
        """Larger cluster of external workers is PHYSICAL."""
        workers = [
            Worker(hostname=f"h{i}", address=f"8.8.8.{i+1}")
            for i in range(10)
        ]
        label = classify(workers, "8.8.8.20")
        assert label == ExecutionLabel.PHYSICAL


class TestClassifyEdgeCases:
    """Boundary conditions and special cases."""

    def test_rfc1918_boundary_10_255_255_255(self) -> None:
        """Highest address in 10.0.0.0/8 is private."""
        workers = [Worker(hostname="h1", address="10.255.255.255")]
        label = classify(workers, "8.8.8.1")
        assert label == ExecutionLabel.LOOPBACK

    def test_rfc1918_boundary_172_31_255_255(self) -> None:
        """Highest address in 172.16.0.0/12 is private."""
        workers = [Worker(hostname="h1", address="172.31.255.255")]
        label = classify(workers, "8.8.8.1")
        assert label == ExecutionLabel.LOOPBACK

    def test_just_outside_rfc1918_172_32(self) -> None:
        """172.32.0.0 (just outside 172.16.0.0/12) is public."""
        workers = [Worker(hostname="h1", address="172.32.0.0")]
        label = classify(workers, "8.8.8.1")
        assert label == ExecutionLabel.PHYSICAL

    def test_just_outside_rfc1918_172_15(self) -> None:
        """172.15.255.255 (just outside 172.16.0.0/12) is public."""
        workers = [Worker(hostname="h1", address="172.15.255.255")]
        label = classify(workers, "8.8.8.1")
        assert label == ExecutionLabel.PHYSICAL

    def test_localhost_string_not_valid(self) -> None:
        """Hostname strings are validated as IP addresses; 'localhost' is not."""
        with pytest.raises(ValueError, match="not a valid IP address"):
            Worker(hostname="h1", address="localhost")
