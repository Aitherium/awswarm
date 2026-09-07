"""Command line over awswarm: sub-layer placement across GPUs too small for a layer.

  assess   will this pool assemble the work, and how many attempts on average
  needed   how many identical volunteers to recruit for a target confidence
  verify   do the replicas agree, and if not, which worker disagreed

Reads JSON from a path or - for stdin; --json for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from . import acquire, integrity


def _load(path: str):
    """Read JSON from a path, or from stdin when the path is '-'."""
    if path == "-":
        return json.load(sys.stdin)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _workers(rows) -> List[acquire.VolunteerWorker]:
    """Build the pool. availability is required, never defaulted to 1.0 --
    assuming full attendance answers a question about a different pool."""
    out = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise SystemExit(f"worker {i}: expected an object, got {type(r).__name__}")
        for key in ("worker_id", "capacity", "availability"):
            if key not in r:
                raise SystemExit(f"worker {i}: missing {key!r}")
        out.append(acquire.VolunteerWorker(
            worker_id=str(r["worker_id"]),
            capacity=int(r["capacity"]),
            availability=float(r["availability"])))
    return out


def cmd_assess(args) -> int:
    report = acquire.assess(args.need, _workers(_load(args.pool)))
    if args.json:
        print(json.dumps({
            "need": args.need,
            "probability": report.probability,
            "expected_attempts": report.expected_attempts,
            "fits_if_all_present": report.fits_if_all_present,
            "summary": report.summary(),
        }, indent=2))
    else:
        print(report.summary())
    # Cannot hold it even at full attendance: exit non-zero so a caller stops retrying.
    return 0 if report.fits_if_all_present else 1


def cmd_needed(args) -> int:
    """Identical volunteers to recruit -- takes a worker TYPE, not a pool."""
    n = acquire.workers_needed(
        args.need, args.capacity, args.availability, args.confidence)
    if n is None:
        print(f"no number of {args.capacity}-unit workers at "
              f"{args.availability:.0%} availability reaches {args.confidence:.0%}")
        return 1
    print(json.dumps({"workers_needed": n}, indent=2) if args.json else
          f"{n} worker(s) of {args.capacity} units at "
          f"{args.availability:.0%} availability, for {args.confidence:.0%} confidence")
    return 0


def cmd_verify(args) -> int:
    import numpy as np

    payload = _load(args.results)
    results = [integrity.ComputationResult(
        tensor=np.asarray(r["tensor"], dtype=float), worker_id=str(r["worker_id"]))
        for r in payload]
    try:
        vote = integrity.aggregate_results(results, tolerance=args.tolerance)
    except Exception as exc:
        # The disagreement names the worker: print it and fail, never average it away.
        print(json.dumps({"verdict": "DISAGREEMENT", "detail": str(exc)}, indent=2)
              if args.json else f"DISAGREEMENT: {exc}")
        return 1
    out = {"verdict": vote.verdict.value, "max_deviation": float(vote.max_deviation)}
    print(json.dumps(out, indent=2) if args.json else
          f"{vote.verdict.value}  max deviation {vote.max_deviation:.3e}")
    return 0 if vote.verdict is integrity.IntegrityVerdict.AGREEMENT else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="awswarm", description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="cmd")

    p_a = sub.add_parser("assess", help="will this pool assemble the work")
    p_a.add_argument("--need", type=int, required=True, help="units required")
    p_a.add_argument("--pool", required=True,
                     help="JSON array of {worker_id, capacity, availability}, or -")

    p_n = sub.add_parser(
        "needed", help="how many identical volunteers to recruit for a confidence")
    p_n.add_argument("--need", type=int, required=True, help="units required")
    p_n.add_argument("--capacity", type=int, required=True,
                     help="units each recruited worker contributes")
    p_n.add_argument("--availability", type=float, required=True,
                     help="probability one is present, 0..1")
    p_n.add_argument("--confidence", type=float, default=0.95)

    p_v = sub.add_parser("verify", help="do the replicas agree, and if not who is wrong")
    p_v.add_argument("--results", required=True,
                     help="JSON array of {worker_id, tensor}, or -")
    p_v.add_argument("--tolerance", type=float, default=1e-6)
    return ap


def main(argv: "list[str] | None" = None) -> int:
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    # GENERATED repo-state intercept (gen_aw_doctor.py) -- do not edit
    try:
        from awgit import state as _aw_state
    except Exception:
        _aw_state = None
    if _aw_state is not None:
        _sv = locals().get("argv")
        if _aw_state.cli_banner(_sv if _sv is not None else __import__("sys").argv[1:]):
            return 0
    args = build_parser().parse_args(argv)
    if not args.cmd:
        build_parser().print_help()
        return 2
    return {"assess": cmd_assess, "needed": cmd_needed, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
