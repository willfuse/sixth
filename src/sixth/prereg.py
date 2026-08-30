"""Preregistration -- borrowed from clinical trials, as the article suggests.

You write down what you expect *before* the test runs, it gets hashed and sealed,
and the verdict is computed by machine against that record. You cannot move the
goalposts on a strategy that preregistered its own thesis.

The seal is a SHA-256 over the canonical JSON of the expectation plus the moment
it was sealed. The database refuses updates to sealed rows (see graph.py's
triggers), so the hash is a check on tampering, not the only defence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

OPS = {
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: abs(a - b) < 1e-12,
    "!=": lambda a, b: abs(a - b) >= 1e-12,
}

# Verdicts a sealed test can return.
CONFIRMED = "confirmed"
REFUTED = "refuted"
INCONCLUSIVE = "inconclusive"
REGIME_CONDITIONAL = "regime_conditional"


@dataclass
class Expectation:
    """What you claim will happen, in machine-checkable form.

    `must` are the conditions that decide confirmed vs refuted. `should` are
    secondary -- failing them alone yields `inconclusive`, not a refutation, but
    they are still recorded and still count against you later.
    """

    must: Dict[str, Dict[str, float]] = field(default_factory=dict)
    should: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rationale: str = ""
    expected_regimes: List[str] = field(default_factory=list)
    horizon: str = ""

    def canonical(self) -> str:
        return json.dumps({
            "must": self.must, "should": self.should, "rationale": self.rationale,
            "expected_regimes": sorted(self.expected_regimes), "horizon": self.horizon,
        }, sort_keys=True, separators=(",", ":"))

    def seal(self, sealed_at: Optional[str] = None) -> Tuple[str, str]:
        """Return (sealed_at_iso, seal_hash)."""
        ts = sealed_at or datetime.now(timezone.utc).isoformat()
        h = hashlib.sha256((self.canonical() + "|" + ts).encode()).hexdigest()
        return ts, h

    def verify(self, sealed_at: str, seal_hash: str) -> bool:
        return self.seal(sealed_at)[1] == seal_hash

    def to_json(self) -> str:
        return self.canonical()

    @classmethod
    def from_json(cls, blob: str) -> "Expectation":
        d = json.loads(blob)
        return cls(must=d.get("must", {}), should=d.get("should", {}),
                   rationale=d.get("rationale", ""),
                   expected_regimes=d.get("expected_regimes", []),
                   horizon=d.get("horizon", ""))

    @classmethod
    def parse(cls, exprs: List[str], rationale: str = "",
              regimes: Optional[List[str]] = None,
              should: Optional[List[str]] = None) -> "Expectation":
        """Build from CLI-friendly strings like "sharpe >= 0.8"."""
        return cls(must=_parse_exprs(exprs), should=_parse_exprs(should or []),
                   rationale=rationale, expected_regimes=regimes or [])


def _parse_exprs(exprs: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for raw in exprs:
        expr = raw.strip()
        for op in (">=", "<=", "==", "!=", ">", "<"):
            if op in expr:
                left, right = expr.split(op, 1)
                metric = left.strip()
                try:
                    value = float(right.strip())
                except ValueError:
                    raise ValueError(f"expectation {raw!r}: {right.strip()!r} is not a number")
                out.setdefault(metric, {})[op] = value
                break
        else:
            raise ValueError(
                f"expectation {raw!r} has no comparison operator (use >=, <=, >, <, ==, !=)")
    return out


@dataclass
class CheckResult:
    metric: str
    op: str
    threshold: float
    actual: Optional[float]
    passed: bool
    tier: str  # "must" | "should"

    def describe(self) -> str:
        got = "missing" if self.actual is None else f"{self.actual:.4f}"
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.metric} {self.op} {self.threshold:g} (got {got})"


def evaluate(exp: Expectation, metrics: Dict[str, float],
             regime_metrics: Optional[Dict[str, Dict[str, float]]] = None
             ) -> Tuple[str, List[CheckResult]]:
    """Score observed metrics against the sealed expectation.

    Returns (verdict, checks). A hypothesis that fails overall but passes its
    `must` conditions inside some regimes comes back `regime_conditional` -- the
    single most useful state in the graph, and the one a log file cannot express.
    """
    checks: List[CheckResult] = []
    for tier, book in (("must", exp.must), ("should", exp.should)):
        for metric, conds in book.items():
            actual = metrics.get(metric)
            for op, threshold in conds.items():
                passed = actual is not None and OPS[op](actual, threshold)
                checks.append(CheckResult(metric, op, threshold, actual, passed, tier))

    must_checks = [c for c in checks if c.tier == "must"]
    should_checks = [c for c in checks if c.tier == "should"]

    if not must_checks:
        return INCONCLUSIVE, checks
    if all(c.passed for c in must_checks):
        return (CONFIRMED if all(c.passed for c in should_checks) else INCONCLUSIVE), checks

    # Failed overall -- but did it hold anywhere?
    if regime_metrics:
        for _, rm in regime_metrics.items():
            if all(OPS[op](rm[m], t)
                   for m, conds in exp.must.items() if m in rm
                   for op, t in conds.items()):
                return REGIME_CONDITIONAL, checks
    return REFUTED, checks


def passing_regimes(exp: Expectation,
                    regime_metrics: Dict[str, Dict[str, float]]) -> List[str]:
    """Which regimes satisfy every `must` condition."""
    out = []
    for label, rm in regime_metrics.items():
        applicable = [(m, op, t) for m, conds in exp.must.items() if m in rm
                      for op, t in conds.items()]
        if applicable and all(OPS[op](rm[m], t) for m, op, t in applicable):
            out.append(label)
    return sorted(out)
