"""Physical-execution labelling must hold in BOTH directions.

A gate that refuses everything passes a one-directional test while having deleted the
feature. So every refusal case here is paired with an acceptance case: genuinely
distributed workers MUST be labelled PHYSICAL, or the module is just an obstacle.

The module shipped without tests at all (its build agent died after writing the source),
so these were written against the stated contract rather than derived from the code —
which is the only way a test can disagree with an implementation.
"""

from __future__ import annotations

import pytest
from awswarm.physical import (
    ExecutionLabel,
    NotPhysicalError,
    Worker,
    classify,
    require_physical,
)

# NOT a documentation range. Python's ipaddress marks RFC 5737 TEST-NET blocks
# (192.0.2/24, 198.51.100/24, 203.0.113/24) as is_private=True, so the addresses you
# instinctively reach for in a test CANNOT stand in for public ones here -- the first
# draft of this file used them and the acceptance case failed, which read as the module
# refusing everything when it was the fixture that was wrong. Pinned by
# test_documentation_ranges_are_private_in_python below so nobody re-loses that hour.
CONTROLLER = "9.9.9.9"


def _distinct_public_pool() -> list[Worker]:
    return [
        Worker("gpu-alpha", "8.8.8.8"),
        Worker("gpu-beta", "1.1.1.1"),
        Worker("gpu-gamma", "8.8.4.4"),
    ]


# --------------------------------------------------------------------------
# The acceptance direction — without this the rest proves nothing
# --------------------------------------------------------------------------


def test_genuinely_distributed_pool_is_physical():
    label = classify(_distinct_public_pool(), CONTROLLER)
    print(f"\n  three distinct public workers -> {label.value}")
    assert label is ExecutionLabel.PHYSICAL
    require_physical(_distinct_public_pool(), CONTROLLER)  # must NOT raise


# --------------------------------------------------------------------------
# The refusal direction
# --------------------------------------------------------------------------


def test_everything_on_one_box_is_loopback_not_physical():
    """The commonest self-deception: it all 'works' on the dev machine."""
    pool = [Worker("localbox-1", "127.0.0.1"), Worker("localbox-2", "127.0.0.2")]
    label = classify(pool, "127.0.0.1")
    print(f"\n  all-loopback -> {label.value}")
    assert label is ExecutionLabel.LOOPBACK
    with pytest.raises(NotPhysicalError) as e:
        require_physical(pool, "127.0.0.1")
    print(f"  refusal: {e.value}")
    assert "loopback" in str(e.value)


@pytest.mark.parametrize(
    "addr,why",
    [
        ("127.0.0.1", "IPv4 loopback"),
        ("::1", "IPv6 loopback — not just 127.0.0.1"),
        ("10.0.0.5", "RFC1918 10/8"),
        ("192.168.1.9", "RFC1918 192.168/16"),
        ("172.16.0.4", "RFC1918 172.16/12"),
        ("169.254.1.1", "link-local"),
    ],
)
def test_non_routable_addresses_are_not_physical(addr, why):
    """One private worker is enough to sink the claim — a chain is as strong as its
    weakest member, and 'mostly distributed' is not a thing."""
    pool = [Worker("real-gpu", "8.8.8.8"), Worker("hidden-gpu", addr)]
    label = classify(pool, CONTROLLER)
    print(f"\n  {why:35} {addr:16} -> {label.value}")
    assert label is not ExecutionLabel.PHYSICAL
    with pytest.raises(NotPhysicalError) as e:
        require_physical(pool, CONTROLLER)
    assert "hidden-gpu" in str(e.value), "the refusal must NAME the offending worker"


def test_worker_at_the_controllers_own_address_is_refused():
    """The controller counting itself as a worker is how a 1-machine run reports as N."""
    pool = [Worker("gpu-alpha", "8.8.8.8"), Worker("gpu-beta", CONTROLLER)]
    with pytest.raises(NotPhysicalError) as e:
        require_physical(pool, CONTROLLER)
    print(f"\n  refusal: {e.value}")
    assert "gpu-beta" in str(e.value) and CONTROLLER in str(e.value)


def test_shared_hostname_is_refused():
    """Two workers on one hostname is one machine wearing two hats."""
    pool = [Worker("same-box", "8.8.8.8"), Worker("same-box", "1.1.1.1")]
    with pytest.raises(ValueError) as e:
        classify(pool, CONTROLLER)
    print(f"\n  refusal: {e.value}")
    assert "same-box" in str(e.value)


def test_simulated_wins_over_every_other_signal():
    """An explicitly simulated run must never be recorded as hardware, however
    convincing its addresses look."""
    label = classify(_distinct_public_pool(), CONTROLLER, simulated=True)
    assert label is ExecutionLabel.SIMULATED
    with pytest.raises(NotPhysicalError) as e:
        require_physical(_distinct_public_pool(), CONTROLLER, simulated=True)
    assert "SIMULATED" in str(e.value)


def test_the_label_cannot_be_supplied_by_the_caller():
    """classify() derives the label from evidence. If a caller could pass one in, the
    whole module would be decorative — so assert the signature offers no such door."""
    import inspect

    params = set(inspect.signature(classify).parameters)
    assert "label" not in params and "execution_label" not in params
    assert params == {"workers", "controller_address", "simulated"}


def test_empty_pool_refuses_rather_than_defaulting():
    """Zero workers is not 'trivially distributed'."""
    with pytest.raises(ValueError):
        classify([], CONTROLLER)


def test_malformed_address_raises_rather_than_being_treated_as_private():
    """A hostname string where an IP belongs must NOT silently fall through to
    'not externally reachable' — that would turn a bug into a quiet LOOPBACK."""
    with pytest.raises(ValueError) as e:
        Worker("gpu", "not-an-ip")
    assert "not a valid IP" in str(e.value)


def test_documentation_ranges_are_private_in_python():
    """The trap this file already fell into, pinned so it costs nobody else an hour.

    RFC 5737 reserves 192.0.2/24, 198.51.100/24 and 203.0.113/24 for documentation, and
    they are the obvious choice for a test that needs a "public" address. Python's
    ipaddress calls them PRIVATE, so a pool built from them classifies LOOPBACK and the
    module looks broken. It is not; the fixture is.
    """
    import ipaddress

    for addr in ("192.0.2.7", "198.51.100.10", "203.0.113.1"):
        assert ipaddress.ip_address(addr).is_private, addr
    for addr in ("8.8.8.8", "1.1.1.1", "9.9.9.9"):
        assert not ipaddress.ip_address(addr).is_private, addr
