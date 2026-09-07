"""Physical execution label enforcement — distinguish real distributed hardware from
simulation and local-only configurations.

WHY THIS EXISTS
---------------------------------------------------------------------------
A run that claims "physical" execution using real GPUs across distinct machines
is not a claim until those machines are proven to exist and be reachable from
outside. The silent failure mode this addresses is simple: claiming a
distributed result that came from a single box or from simulated workers reads
as real evidence until the moment someone tries to verify it.

This module answers one question: given a controller address and a list of
workers with their addresses and hostnames, is this deployment REALLY spread
across independent, externally-reachable machines, or is it a LOOPBACK
(everything on the same host or all loopback addresses), or is it SIMULATED
(explicitly marked as such)?

WHAT IS PROVEN HERE, TODAY, ON THIS MACHINE
---------------------------------------------------------------------------
A deployment is labeled PHYSICAL if and only if:
- Every worker has a distinct hostname (no worker is the same host as another)
- Every worker's address is NOT in RFC1918 private ranges, IPv6 loopback, or
  IPv4 loopback space
- At least one worker address is distinct from the controller's address

The proofs are in tests/test_physical.py, which verifies both directions:
the function correctly labels genuinely distributed setups AND correctly
refuses setups that are co-located or private.

WHAT IS NOT CLAIMED
---------------------------------------------------------------------------
- This does not test whether addresses are actually reachable (use a network
  probe for that).
- This does not verify that claimed "external" addresses are not behind NAT
  (that is a problem for the operator, not for this function).
- This does not distinguish between different types of private addresses
  (RFC1918 rules, link-local, etc.) — it simply rejects ALL of them as
  non-physical.
- This does not cache or remember prior classifications: every call derives
  the label from evidence, fresh.
- This does not accept caller-supplied labels; the label is ALWAYS derived
  from the workers and controller address you pass.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from enum import Enum
from typing import List


class ExecutionLabel(Enum):
    """Classification of where a run's work truly executes.

    SIMULATED
        The run is explicitly marked as using simulated workers and has no
        claim on real distributed hardware.
    LOOPBACK
        All workers and the controller are on the same physical machine or
        use only loopback addressing. Everything is co-located.
    PHYSICAL
        Workers are on distinct physical machines with externally-reachable
        addresses, and the deployment is genuinely distributed.
    """

    SIMULATED = "simulated"
    LOOPBACK = "loopback"
    PHYSICAL = "physical"


@dataclass(frozen=True)
class Worker:
    """One member of the execution pool — its address and identity.

    hostname
        The machine's name, used to detect co-location. Two workers on the
        same hostname are on the same physical machine.
    address
        A string IP address (IPv4 or IPv6) used to detect loopback and private
        ranges. Must be parseable by `ipaddress.ip_address()`.
    """

    hostname: str
    address: str

    def __post_init__(self) -> None:
        if not self.hostname or not self.hostname.strip():
            raise ValueError("hostname must not be empty or whitespace-only")
        if not self.address or not self.address.strip():
            raise ValueError("address must not be empty or whitespace-only")
        # Validate that address is parseable by ipaddress.
        try:
            ipaddress.ip_address(self.address)
        except ValueError as e:
            raise ValueError(
                f"address '{self.address}' is not a valid IP address: {e}"
            ) from e


def _is_externally_reachable(addr_str: str) -> bool:
    """True if the IP address is neither loopback, link-local, nor private.

    Externally reachable means the address can (in principle) be reached from
    outside the local network. This is a necessary but NOT sufficient condition
    for physical execution — the address could still be behind NAT — but a
    loopback or private address is DEFINITELY not reachable from outside.

    All RFC1918 private ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16),
    IPv6 loopback (::1), IPv4 loopback (127.0.0.0/8), and link-local addresses
    are rejected.
    """
    addr = ipaddress.ip_address(addr_str)

    # Loopback: localhost
    if addr.is_loopback:
        return False

    # Private (RFC1918): 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
    if addr.is_private:
        return False

    # Link-local: 169.254.0.0/16 (IPv4), fe80::/10 (IPv6)
    if addr.is_link_local:
        return False

    return True


def classify(
    workers: List[Worker],
    controller_address: str,
    simulated: bool = False,
) -> ExecutionLabel:
    """Derive the execution label from workers and controller address.

    Answers the question: are all these workers on truly distinct, externally-
    reachable machines? Or are they co-located (LOOPBACK)? Or is this a
    simulated run?

    The label is DERIVED, not accepted from the caller. The function looks at
    the evidence — hostnames, addresses — and tells you what the deployment
    really is.

    Arguments:
        workers: List of Worker(hostname, address) tuples. Empty list raises
            ValueError (cannot claim physical distribution with zero workers).
        controller_address: The orchestrator's IP address, used to detect if
            workers are on the same machine as the controller. Must be a valid
            IP address string.
        simulated: If True, all other evidence is ignored and SIMULATED is
            returned immediately. Explicit simulation always wins.

    Returns:
        ExecutionLabel.SIMULATED, LOOPBACK, or PHYSICAL.

    Raises:
        ValueError: if workers is empty, if any Worker is invalid, if
            controller_address is not a valid IP, or if two workers claim the
            same hostname (a single physical machine cannot run two distinct
            workers).
    """
    if simulated:
        return ExecutionLabel.SIMULATED

    if not workers:
        raise ValueError("no workers given — cannot classify distribution with zero workers")

    # Validate controller address.
    try:
        ipaddress.ip_address(controller_address)
    except ValueError as e:
        raise ValueError(
            f"controller_address '{controller_address}' is not a valid IP address: {e}"
        ) from e

    # Collect all hostnames and detect duplicates.
    hostnames = [w.hostname for w in workers]
    if len(hostnames) != len(set(hostnames)):
        duplicates = [h for h in set(hostnames) if hostnames.count(h) > 1]
        raise ValueError(
            f"workers cannot share a hostname (single machine, not distributed): "
            f"{duplicates}"
        )

    # Check if all workers are externally reachable.
    externally_reachable = [
        _is_externally_reachable(w.address) for w in workers
    ]

    if not all(externally_reachable):
        # At least one worker is in loopback or private space.
        return ExecutionLabel.LOOPBACK

    # All workers are externally reachable. Check if any is distinct from the
    # controller.
    worker_addresses = {w.address for w in workers}
    if controller_address in worker_addresses:
        # Controller is co-located with at least one worker. Still loopback.
        return ExecutionLabel.LOOPBACK

    # All workers are externally reachable and distinct from the controller.
    return ExecutionLabel.PHYSICAL


class NotPhysicalError(Exception):
    """A run was claimed PHYSICAL and the evidence does not support it."""


def require_physical(
    workers: List[Worker],
    controller_address: str,
    simulated: bool = False,
) -> None:
    """Refuse to let a run be recorded as PHYSICAL unless the evidence says so.

    `classify` REPORTS; this REFUSES, and the difference is the whole point. A caller
    that has to read a returned label can ignore it, and the failure mode being guarded
    against is precisely a run recorded as real hardware when it was not. A label
    enforced by prose is enforced by nobody, so this raises.

    The message names the offender -- which worker, which address, why -- because
    "not physical" alone sends whoever reads it back to re-derive what this function
    already computed.
    """
    label = classify(workers, controller_address, simulated=simulated)
    if label is ExecutionLabel.PHYSICAL:
        return

    if label is ExecutionLabel.SIMULATED:
        raise NotPhysicalError(
            "run is explicitly SIMULATED -- it cannot be recorded as physical hardware"
        )

    unreachable = [w for w in workers if not _is_externally_reachable(w.address)]
    if unreachable:
        raise NotPhysicalError(
            "not PHYSICAL: "
            + ", ".join(
                f"{w.hostname} at {w.address} is loopback/private, not externally "
                "reachable" for w in unreachable
            )
        )
    colocated = [w for w in workers if w.address == controller_address]
    if colocated:
        raise NotPhysicalError(
            "not PHYSICAL: "
            + ", ".join(
                f"{w.hostname} shares the controller's address {controller_address}"
                for w in colocated
            )
        )
    raise NotPhysicalError(f"not PHYSICAL: classified {label.value} on the given evidence")
