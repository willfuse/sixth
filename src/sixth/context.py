"""Render the graph as context for a coding agent.

This is where the loop actually closes. The graph is only worth keeping if the
thing that writes the next strategy reads it first -- otherwise it is an archive,
not a memory.

`brief()` emits a markdown block to paste into Claude Code (or any agent) before
asking for the next strategy. `payload()` is the same content as JSON for
programmatic use. Both are deliberately compact: an agent that gets 40kb of prior
results ignores all of it, so this ranks hard and truncates.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .graph import HypothesisGraph
from .prereg import CONFIRMED, REFUTED, REGIME_CONDITIONAL


def payload(graph: HypothesisGraph, limit: int = 12,
            focus: Optional[str] = None) -> Dict[str, Any]:
    """Structured prior knowledge: what failed, what held, what is untried."""
    refuted = []
    for h in graph.all(status=REFUTED):
        exps = graph.experiments(h.id)
        last = exps[-1] if exps else None
        refuted.append({
            "slug": h.slug, "statement": h.statement,
            "sharpe": round(last.metrics.get("sharpe", 0.0), 3) if last else None,
            "why": [l["text"] for l in graph.lessons(h.id)][:3],
        })

    conditional = []
    for h, regimes in graph.confirmed_only_in():
        conditional.append({
            "slug": h.slug, "statement": h.statement, "works_in": regimes,
            "fails_in": sorted({l["regime"] for l in graph.lessons(h.id)
                                if l["kind"] == "regime" and l["regime"]
                                and "Loses money" in l["text"]}),
        })

    confirmed = [{"slug": h.slug, "statement": h.statement}
                 for h in graph.all(status=CONFIRMED)]

    frontier = [{"slug": h.slug, "statement": h.statement, "tags": h.tags}
                for h in graph.never_tried()]

    if focus:
        rank = {h.slug: s for h, s in graph.similar(focus, k=100)}
        key = lambda d: -rank.get(d["slug"], 0.0)
        refuted.sort(key=key)
        conditional.sort(key=key)
        frontier.sort(key=key)

    # Cross-cutting lessons: the patterns that hold regardless of hypothesis.
    counts: Dict[str, int] = {}
    for l in graph.lessons():
        counts[l["kind"]] = counts.get(l["kind"], 0) + 1
    recurring = [
        {"kind": l["kind"], "text": l["text"], "regime": l["regime"]}
        for l in graph.lessons()
        if counts.get(l["kind"], 0) >= 2
    ][:limit]

    return {
        "summary": graph.summary(),
        "confirmed": confirmed[:limit],
        "regime_conditional": conditional[:limit],
        "refuted": refuted[:limit],
        "frontier": frontier[:limit],
        "recurring_lessons": recurring,
        "trials_run": graph.trial_count(),
    }


def brief(graph: HypothesisGraph, limit: int = 12,
          focus: Optional[str] = None) -> str:
    """Markdown block for an agent's prompt. Paste this above the request."""
    p = payload(graph, limit, focus)
    s = p["summary"]
    L: List[str] = []
    L.append("## Prior research state (do not re-derive)")
    L.append("")
    L.append(f"This graph holds {s['hypotheses']} hypotheses across "
             f"{s['experiments']} recorded experiments and {p['trials_run']} total trials. "
             f"Treat everything below as already established. Do not propose anything "
             f"listed under REFUTED without an explicit reason the earlier failure "
             f"does not apply.")
    L.append("")

    if p["confirmed"]:
        L.append("### Confirmed")
        for h in p["confirmed"]:
            L.append(f"- **{h['slug']}** — {h['statement']}")
        L.append("")

    if p["regime_conditional"]:
        L.append("### Works only in specific regimes")
        for h in p["regime_conditional"]:
            line = f"- **{h['slug']}** — {h['statement']}"
            if h["works_in"]:
                line += f"\n  - holds in: {', '.join(h['works_in'])}"
            if h["fails_in"]:
                line += f"\n  - bleeds in: {', '.join(h['fails_in'])}"
            L.append(line)
        L.append("")

    if p["refuted"]:
        L.append("### Refuted — already tried, already failed")
        for h in p["refuted"]:
            sr = f" (Sharpe {h['sharpe']})" if h["sharpe"] is not None else ""
            L.append(f"- **{h['slug']}** — {h['statement']}{sr}")
            for w in h["why"]:
                L.append(f"  - {w}")
        L.append("")

    if p["recurring_lessons"]:
        L.append("### Recurring failure patterns in this research programme")
        for l in p["recurring_lessons"]:
            tag = f"[{l['kind']}]" + (f"[{l['regime']}]" if l["regime"] else "")
            L.append(f"- {tag} {l['text']}")
        L.append("")

    if p["frontier"]:
        L.append("### Frontier — stated but never tested")
        for h in p["frontier"]:
            L.append(f"- **{h['slug']}** — {h['statement']}")
        L.append("")

    L.append("### Rules for the next proposal")
    L.append("1. It must not duplicate anything under Refuted, or repeat a listed "
             "recurring failure pattern.")
    L.append("2. State it as a falsifiable claim with numeric must-conditions "
             "(e.g. `sharpe >= 0.8`, `max_drawdown <= 0.25`) so it can be sealed "
             "before the test runs.")
    L.append(f"3. Assume {p['trials_run']} trials have already been spent on this "
             "programme; the Deflated Sharpe bar rises with every additional one.")
    return "\n".join(L)


def as_json(graph: HypothesisGraph, limit: int = 12,
            focus: Optional[str] = None) -> str:
    return json.dumps(payload(graph, limit, focus), indent=2)
