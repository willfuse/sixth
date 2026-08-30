"""Driving the loop from an agent, rather than from the CLI.

The pattern: read the graph, propose something it has never tried, seal an
expectation, run, and let the result write itself back. The only step that needs
a model is the proposal -- everything else is deterministic and lives here.

Run:  python examples/agent_loop.py
"""

from sixth import (Costs, Expectation, HypothesisGraph, brief, render_cycle,
                   run_cycle, synthetic_bars)

GRAPH = "research.sqlite"


def propose_with_an_agent(context_block: str) -> dict:
    """Where you would call Claude, an API, or your own head.

    The contract: hand it `context_block` (everything already learned) and get
    back a falsifiable claim plus the numeric conditions that would settle it.
    Stubbed here so the example runs with no key and no network.
    """
    _ = context_block  # in a real implementation, this goes into the prompt
    return {
        "statement": "A 20/100 MA cross beats buy-and-hold, but flat in downtrends.",
        "tags": ["trend", "regime-filtered"],
        "strategy": "sma_cross",
        "grid": {"fast": [10, 20, 40], "slow": [60, 100, 200]},
        "must": ["sharpe >= 0.8", "max_drawdown <= 0.25"],
        "rationale": "The unfiltered version bled in every down regime; sitting "
                     "out downtrends should keep the trend capture and drop the bleed.",
    }


def main() -> None:
    graph = HypothesisGraph(GRAPH)
    bars = synthetic_bars(n=2520)

    # 1. Everything already known goes to the agent. This is the closed loop:
    #    without it, the proposal below would be indistinguishable from the one
    #    made before the last forty experiments.
    context_block = brief(graph, focus="trend following")
    proposal = propose_with_an_agent(context_block)

    # 2. Refuse to re-run something the graph has already settled.
    near = graph.similar(proposal["statement"], k=1)
    if near and near[0][1] > 0.92:
        h, score = near[0]
        print(f"already tested as {h.slug} ({h.status}, similarity {score:.2f}) -- skipping")
        return

    # 3. State it, then seal the expectation BEFORE anything runs.
    h = graph.add(proposal["statement"], tags=proposal["tags"])
    graph.preregister(h.id, Expectation.parse(proposal["must"],
                                              rationale=proposal["rationale"]))

    # 4. Run the cycle. The result, the regime breakdown, the lessons and the
    #    follow-up hypotheses all land in the graph on their own.
    result = run_cycle(graph, h.id, proposal["strategy"], bars,
                       grid=proposal["grid"], costs=Costs(),
                       create_proposals=True)
    print(render_cycle(result))

    print("\n" + "=" * 68)
    print("The next agent starts here:\n")
    print(brief(graph))


if __name__ == "__main__":
    main()
