"""ROADMAP.md is parsed and asserted, because prose rots.

A roadmap is the one document with a standing incentive to drift optimistic: statuses
age toward PASS as memory of what was actually measured fades, and nothing complains.
This repo's whole gate family exists because a rule nothing asserts is a suggestion.

The load-bearing arm is `test_no_gate_claims_pass_without_naming_evidence`. Everything
else here is structure — that one is the honesty check, and it is the reason this file
exists rather than a note asking people to keep the roadmap current.

Written after the build agent that produced ROADMAP.md died before writing its test, so
the doc shipped unasserted for a while — which is exactly the state it describes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROADMAP = Path(__file__).resolve().parents[1] / "docs" / "ROADMAP.md"

EXPECTED_GATES = 8

#: A status must be one of these. `PASS` may carry a parenthetical narrowing
#: ("PASS (protocol only)") — that narrowing is a FEATURE: an unqualified PASS on a
#: half-met gate is the overclaim this file is guarding against, so the vocabulary
#: deliberately makes the honest form easy to write.
ALLOWED_STATUS_PREFIXES = ("PASS", "PARTIAL", "NOT MET")

#: A PASS must point at something a reader can go and check. Prose saying it works is
#: what a roadmap says on its way to being wrong.
EVIDENCE_MARKERS = (".py", "tests/", "measured", "Measured", "printed", "commit")


def _text() -> str:
    assert ROADMAP.exists(), f"ROADMAP.md missing at {ROADMAP}"
    return ROADMAP.read_text(encoding="utf-8", errors="replace")


def _gate_blocks() -> list[tuple[str, str]]:
    """Return [(heading, body)] for each '## Gate N: ...' section."""
    text = _text()
    parts = re.split(r"^## (Gate \d+:[^\n]*)$", text, flags=re.M)
    # parts[0] is the preamble; then alternating heading, body
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def test_all_eight_gates_are_present_and_numbered_once():
    blocks = _gate_blocks()
    print(f"\n  gates found: {len(blocks)}")
    for h, _ in blocks:
        print(f"    {h}")
    assert len(blocks) == EXPECTED_GATES
    numbers = sorted(int(re.match(r"Gate (\d+):", h).group(1)) for h, _ in blocks)
    assert numbers == list(range(1, EXPECTED_GATES + 1)), f"gate numbering is {numbers}"


@pytest.mark.parametrize("idx", range(EXPECTED_GATES))
def test_each_gate_has_the_required_sections(idx):
    heading, body = _gate_blocks()[idx]
    for section in ("**What it asks**", "**Evidence for PASS**", "**Status TODAY**"):
        assert section in body, f"{heading} is missing {section}"


@pytest.mark.parametrize("idx", range(EXPECTED_GATES))
def test_each_status_is_from_the_allowed_vocabulary(idx):
    heading, body = _gate_blocks()[idx]
    m = re.search(r"\*\*Status TODAY\*\*:\s*(.+)", body)
    assert m, f"{heading} has no parseable status"
    status = m.group(1).strip()
    assert status.startswith(ALLOWED_STATUS_PREFIXES), (
        f"{heading} status {status!r} is outside {ALLOWED_STATUS_PREFIXES} — an ad-hoc "
        "status is how a roadmap stops being comparable to itself over time"
    )


@pytest.mark.parametrize("idx", range(EXPECTED_GATES))
def test_no_gate_claims_pass_without_naming_evidence(idx):
    """THE arm. An unevidenced PASS must fail the suite.

    A gate that says PASS and points at nothing is indistinguishable from a gate nobody
    checked — and it is strictly worse, because it stops anyone else looking.
    """
    heading, body = _gate_blocks()[idx]
    status = re.search(r"\*\*Status TODAY\*\*:\s*(.+)", body).group(1).strip()
    if not status.startswith("PASS"):
        return
    after = body.split("**Status TODAY**", 1)[1]
    named = [mk for mk in EVIDENCE_MARKERS if mk in after]
    print(f"\n  {heading}: {status} — evidence markers {named}")
    assert named, (
        f"{heading} claims {status!r} and names no file, test or measurement after it. "
        "Point at something a reader can check, or lower the status."
    )


def test_the_roadmap_states_what_awswarm_does_not_do():
    """Scope honesty, asserted. The package is client-shaped math and protocol; a reader
    who skims must still be told there is no live fleet controller and no reproduction of
    any prior run's numbers, because that is precisely what a roadmap implies by omission.
    """
    text = _text().lower()
    # Check the PROPERTY, not a spelling. The first draft demanded the literal string
    # "out-of-scope" and failed on a roadmap that says "separate work" -- a test that
    # dictates vocabulary makes the doc worse, not more honest.
    scope_phrases = (
        "out-of-scope", "out of scope", "not in scope", "separate work",
        "does not", "is not", "no orchestrator", "not here",
    )
    named = [p for p in scope_phrases if p in text]
    print(f"\n  scope-limiting phrases present: {named}")
    assert named, f"ROADMAP.md states no limit on scope; expected one of {scope_phrases}"
    assert "controller" in text or "orchestrator" in text, (
        "the roadmap must say something about the orchestrator it does NOT include"
    )


def test_every_gate_names_the_modules_that_move_it():
    """A gate with no module attribution cannot be re-checked when the code changes."""
    missing = [h for h, b in _gate_blocks() if "**Modules contributing**" not in b]
    assert not missing, f"gates with no module attribution: {missing}"
