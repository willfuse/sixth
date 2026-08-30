"""Command line interface.

    sixth init                     create a graph
    sixth add "<statement>"        state a hypothesis (does not run anything)
    sixth prereg <slug> --must ... seal an expectation
    sixth run <slug> --strategy X  run the full cycle and write the result
    sixth context                  render the graph as agent context
    sixth frontier                 what has never been tried
    sixth show <slug>              one hypothesis, everything known about it
    sixth demo                     the whole loop, end to end, on synthetic data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import context as ctx
from .backtest import Costs
from .data import Bars, load_csv, synthetic_bars
from .graph import EDGE_KINDS, STATUSES, HypothesisGraph
from .loop import render_cycle, run_cycle
from .postmortem import render as render_pm
from .prereg import Expectation
from .strategy import REGISTRY

DEFAULT_DB = os.environ.get("SIXTH_DB", "sixth.sqlite")


# --------------------------------------------------------------------------
def _bars(args: argparse.Namespace) -> Bars:
    if getattr(args, "csv", None):
        return load_csv(args.csv, symbol=getattr(args, "symbol", None) or None)
    return synthetic_bars(symbol=getattr(args, "symbol", None) or "SYNTH",
                          n=args.bars, seed=args.seed)


def _grid(pairs: Optional[Sequence[str]]) -> Dict[str, List[Any]]:
    """Parse `--grid fast=10,20,40 slow=60,100` into a search space."""
    grid: Dict[str, List[Any]] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--grid expects key=v1,v2 (got {pair!r})")
        key, values = pair.split("=", 1)
        parsed: List[Any] = []
        for v in values.split(","):
            v = v.strip()
            try:
                parsed.append(int(v))
            except ValueError:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(v)
        grid[key.strip()] = parsed
    return grid


def _costs(args: argparse.Namespace) -> Costs:
    return Costs(commission_bps=args.commission_bps, slippage_bps=args.slippage_bps,
                 spread_bps=args.spread_bps)


def _open(args: argparse.Namespace) -> HypothesisGraph:
    return HypothesisGraph(args.db)


# --------------------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    g = _open(args)
    print(f"graph ready at {g.path}")
    print(json.dumps(g.summary(), indent=2))
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    g = _open(args)
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    if not args.force:
        near = g.similar(args.statement, k=3)
        strong = [(h, s) for h, s in near if s > 0.75]
        if strong:
            print("This looks like something already in the graph:\n")
            for h, s in strong:
                print(f"  [{s:.2f}] {h.slug}  ({h.status})")
                print(f"         {h.statement}")
            print("\nAdd it anyway with --force, or refine the existing one instead.")
            return 2
    h = g.add(args.statement, tags=tags, parent=args.parent,
              edge_kind=args.edge_kind)
    print(f"added {h.slug}  [{h.status}]")
    if args.parent:
        print(f"  {args.edge_kind} -> {args.parent}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    g = _open(args)
    rows = g.all(status=args.status, tag=args.tag)
    if not rows:
        print("no hypotheses match")
        return 0
    for h in rows:
        n = len(g.experiments(h.id))
        tags = f"  #{','.join(h.tags)}" if h.tags else ""
        print(f"{h.status:<20} {h.slug:<44} {n:>3} runs{tags}")
        print(f"{'':<20} {h.statement}")
    return 0


def cmd_prereg(args: argparse.Namespace) -> int:
    g = _open(args)
    try:
        exp = Expectation.parse(args.must, rationale=args.rationale,
                                regimes=[r.strip() for r in (args.regimes or "").split(",")
                                         if r.strip()],
                                should=args.should)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not exp.must:
        print("error: at least one --must condition is required; an expectation "
              "with nothing falsifiable in it is not an expectation",
              file=sys.stderr)
        return 2
    pid = g.preregister(args.slug, exp)
    sealed_at, seal = exp.seal(g.conn.execute(
        "SELECT sealed_at FROM preregs WHERE id=?", (pid,)).fetchone()[0])
    print(f"sealed prereg #{pid} for {args.slug}")
    for metric, conds in exp.must.items():
        for op, v in conds.items():
            print(f"  must:   {metric} {op} {v:g}")
    for metric, conds in exp.should.items():
        for op, v in conds.items():
            print(f"  should: {metric} {op} {v:g}")
    print(f"  seal:   {seal[:32]}...")
    print("\nThis expectation is now immutable. The database will refuse to edit it.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    g = _open(args)
    if not g.exists(args.slug):
        print(f"error: no hypothesis {args.slug!r}. Add it first with `sixth add`.",
              file=sys.stderr)
        return 2
    bars = _bars(args)
    try:
        c = run_cycle(g, args.slug, args.strategy, bars, _grid(args.grid), _costs(args),
                      holdout=args.holdout, n_folds=args.folds, embargo=args.embargo,
                      min_train=args.min_train, null_samples=args.null_samples,
                      create_proposals=args.create_proposals)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(c.to_dict(), indent=2, default=str))
    else:
        print(render_cycle(c))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    g = _open(args)
    h = g.get(args.slug)
    print(f"{h.slug}  [{h.status}]")
    print(f"  {h.statement}")
    if h.tags:
        print(f"  tags: {', '.join(h.tags)}")
    chain = g.ancestry(h.id)
    if chain:
        print(f"  derived from: {' <- '.join(x.slug for x in chain)}")
    for d, kind, other in g.neighbors(h.id):
        arrow = "->" if d == "out" else "<-"
        print(f"  {arrow} {kind}: {other.slug} [{other.status}]")

    exps = g.experiments(h.id)
    print(f"\n{len(exps)} experiment(s)")
    for e in exps:
        m = e.metrics
        print(f"  #{e.id} {e.created_at[:19]}  {e.verdict.upper()}")
        print(f"     sharpe {m.get('sharpe', 0):.3f}  dsr {m.get('dsr', 0):.3f}  "
              f"maxdd {m.get('max_drawdown', 0):.1%}  trials {int(m.get('n_trials', e.n_trials))}")
        if e.regimes:
            worst = min(e.regimes, key=lambda k: e.regimes[k]["sharpe"])
            best = max(e.regimes, key=lambda k: e.regimes[k]["sharpe"])
            print(f"     best {best} ({e.regimes[best]['sharpe']:.2f})  "
                  f"worst {worst} ({e.regimes[worst]['sharpe']:.2f})")

    ls = g.lessons(h.id)
    if ls:
        print(f"\n{len(ls)} lesson(s)")
        for l in ls:
            tag = f"[{l['kind']}]" + (f"[{l['regime']}]" if l["regime"] else "")
            print(f"  {tag} {l['text']}")
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    g = _open(args)
    if args.json:
        print(ctx.as_json(g, args.limit, args.focus))
    else:
        print(ctx.brief(g, args.limit, args.focus))
    return 0


def cmd_frontier(args: argparse.Namespace) -> int:
    g = _open(args)
    rows = g.never_tried()
    if not rows:
        print("frontier is empty -- every stated hypothesis has been tested")
        return 0
    print(f"{len(rows)} hypothesis/hypotheses never tested\n")
    for h in rows:
        print(f"  {h.slug}")
        print(f"    {h.statement}")
        if h.tags:
            print(f"    tags: {', '.join(h.tags)}")
    return 0


def cmd_lessons(args: argparse.Namespace) -> int:
    g = _open(args)
    rows = g.lessons(kind=args.kind)
    if not rows:
        print("no lessons recorded yet")
        return 0
    for l in rows:
        tag = f"[{l['kind']}]" + (f"[{l['regime']}]" if l["regime"] else "")
        print(f"{tag} {l['text']}")
    return 0


def cmd_regime(args: argparse.Namespace) -> int:
    g = _open(args)
    rows = g.refuted_in(args.regime)
    if not rows:
        print(f"nothing is known to lose money in {args.regime}")
        return 0
    print(f"{len(rows)} hypothesis/hypotheses lose money in {args.regime}\n")
    for h, m in rows:
        print(f"  {h.slug}  sharpe {m.get('sharpe', 0):.2f} over {int(m.get('n', 0))} bars")
        print(f"    {h.statement}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    g = _open(args)
    s = g.summary()
    print(f"graph: {s['path']}")
    print(f"  hypotheses  {s['hypotheses']}")
    for k, v in s["by_status"].items():
        if v:
            print(f"    {k:<20} {v}")
    print(f"  experiments {s['experiments']}")
    print(f"  trials      {s['trials']}   <- the Deflated Sharpe denominator")
    print(f"  lessons     {s['lessons']}")
    print(f"  edges       {s['edges']}")
    print(f"  preregs     {s['preregs']} ({s['open_preregs']} open)")
    if s["seal_failures"]:
        print(f"  !! {s['seal_failures']} preregistration(s) failed their seal check")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    print(json.dumps(_open(args).export(), indent=2, default=str))
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    for name, s in sorted(REGISTRY.items()):
        params = ", ".join(f"{k}={v}" for k, v in s.params.items()) or "-"
        print(f"  {name:<18} {s.description}")
        print(f"  {'':<18} params: {params}")
    return 0


def cmd_link(args: argparse.Namespace) -> int:
    g = _open(args)
    g.link(args.src, args.dst, args.kind, args.note)
    print(f"{args.src} --{args.kind}--> {args.dst}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """The whole argument, executed: state, seal, test, learn, propose."""
    db = args.db
    if os.path.exists(db) and not args.keep:
        os.remove(db)
    g = HypothesisGraph(db)
    bars = synthetic_bars(n=args.bars, seed=args.seed)

    plan = [
        ("A 20/100 moving-average cross captures trend persistence and beats "
         "buy-and-hold risk-adjusted.",
         ["trend"], "sma_cross", {"fast": [10, 20, 40], "slow": [60, 100, 200]},
         ["sharpe >= 0.8", "max_drawdown <= 0.25"],
         "Trends persist long enough that a lagging filter still captures them."),
        ("Fading moves beyond one trailing standard deviation is profitable "
         "intraday-to-daily.",
         ["mean-reversion"], "mean_reversion", {"window": [10, 20, 40], "z": [0.5, 1.0, 2.0]},
         ["sharpe >= 0.5", "max_drawdown <= 0.30"],
         "Short-horizon overreaction reverts."),
        ("Simple 60-day time-series momentum earns a positive Sharpe net of costs.",
         ["trend", "momentum"], "momentum", {"lookback": [20, 60, 120]},
         ["sharpe >= 0.5"],
         "Cross-sectional momentum is well documented; test the time-series form."),
    ]

    for statement, tags, strat, grid, must, why in plan:
        h = g.add(statement, tags=tags)
        g.preregister(h.id, Expectation.parse(must, rationale=why))
        print("\n" + "#" * 68)
        print(f"# {h.slug}")
        print("#" * 68)
        c = run_cycle(g, h.id, strat, bars, grid, Costs(), null_samples=100,
                      create_proposals=True)
        print(render_cycle(c))

    print("\n" + "#" * 68)
    print("# What the graph now knows -- this is what gets handed to the next agent")
    print("#" * 68 + "\n")
    print(ctx.brief(g))
    print("\n" + "-" * 68)
    cmd_status(argparse.Namespace(db=db))
    print(f"\nThe graph persists at {db}. Run `sixth --db {db} context` any time to "
          f"regenerate the block above.")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sixth",
        description="A persistent hypothesis graph for quant research: the part of "
                    "the loop that remembers what already failed.")
    p.add_argument("--db", default=DEFAULT_DB, help=f"graph file (default {DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--csv", help="OHLCV csv; omit to use synthetic bars")
        sp.add_argument("--symbol", default="", help="symbol label")
        sp.add_argument("--bars", type=int, default=2520, help="synthetic bar count")
        sp.add_argument("--seed", type=int, default=7, help="synthetic seed")

    def cost_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--commission-bps", type=float, default=1.0)
        sp.add_argument("--slippage-bps", type=float, default=2.0)
        sp.add_argument("--spread-bps", type=float, default=0.0)

    sp = sub.add_parser("init", help="create or inspect the graph")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("add", help="state a hypothesis")
    sp.add_argument("statement")
    sp.add_argument("--tags", default="", help="comma separated")
    sp.add_argument("--parent", help="slug this one derives from")
    sp.add_argument("--edge-kind", default="derived_from", choices=EDGE_KINDS)
    sp.add_argument("--force", action="store_true",
                    help="add even if a near-duplicate exists")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("list", help="list hypotheses")
    sp.add_argument("--status", choices=STATUSES)
    sp.add_argument("--tag")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("prereg", help="seal an expectation before testing")
    sp.add_argument("slug")
    sp.add_argument("--must", action="append", default=[],
                    metavar="'sharpe >= 0.8'",
                    help="condition that decides confirmed vs refuted (repeatable)")
    sp.add_argument("--should", action="append", default=[],
                    help="secondary condition (repeatable)")
    sp.add_argument("--rationale", default="", help="why you expect this")
    sp.add_argument("--regimes", default="", help="regimes you expect it to hold in")
    sp.set_defaults(func=cmd_prereg)

    sp = sub.add_parser("run", help="run the full cycle against a sealed expectation")
    sp.add_argument("slug")
    sp.add_argument("--strategy", required=True, help="see `sixth strategies`")
    sp.add_argument("--grid", nargs="*", metavar="key=v1,v2",
                    help="parameter search space; every combination counts as a trial")
    sp.add_argument("--holdout", type=float, default=0.2)
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--embargo", type=int, default=10)
    sp.add_argument("--min-train", type=int, default=250)
    sp.add_argument("--null-samples", type=int, default=200)
    sp.add_argument("--create-proposals", action="store_true",
                    help="write the implied follow-up hypotheses into the graph")
    sp.add_argument("--json", action="store_true")
    data_args(sp)
    cost_args(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("show", help="everything known about one hypothesis")
    sp.add_argument("slug")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("context", help="render the graph as agent context")
    sp.add_argument("--limit", type=int, default=12)
    sp.add_argument("--focus", help="rank by relevance to this idea")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_context)

    sp = sub.add_parser("frontier", help="hypotheses never tested")
    sp.set_defaults(func=cmd_frontier)

    sp = sub.add_parser("lessons", help="every lesson recorded")
    sp.add_argument("--kind")
    sp.set_defaults(func=cmd_lessons)

    sp = sub.add_parser("regime", help="what is known to lose money in a regime")
    sp.add_argument("regime", help="e.g. down/stressed")
    sp.set_defaults(func=cmd_regime)

    sp = sub.add_parser("link", help="connect two hypotheses")
    sp.add_argument("src")
    sp.add_argument("dst")
    sp.add_argument("--kind", default="refines", choices=EDGE_KINDS)
    sp.add_argument("--note", default="")
    sp.set_defaults(func=cmd_link)

    sp = sub.add_parser("status", help="graph summary")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("export", help="dump the whole graph as JSON")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("strategies", help="built-in strategies")
    sp.set_defaults(func=cmd_strategies)

    sp = sub.add_parser("demo", help="run the entire loop end to end on synthetic data")
    sp.add_argument("--bars", type=int, default=2520)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--keep", action="store_true", help="append to an existing graph")
    sp.set_defaults(func=cmd_demo)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
