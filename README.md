# sixth

**The part of the quant research loop that remembers.**

Zero dependencies. Pure standard library. One SQLite file.

---

## The idea

A hedge fund is one cycle repeating:

> **Research** → **Code** → **Backtest** → **Live** → **Post-mortem** → **Fine-tune**

Five of those six work today. You can describe a strategy in English and have an
agent write it, backtest it, and push it to a broker in about ninety seconds.

The sixth doesn't work, and it isn't hard — it's just missing. The lesson from a
losing test lands in a log file and stays there. The next strategy your agent
writes has never heard of it. Every cycle starts from roughly the same place, so
nothing compounds.

`sixth` is that missing part: a persistent, queryable graph of what you tested,
what failed, **in which market regime**, and what you have never tried at all.
Negative results are the most undervalued asset in quant research, and almost no
system keeps them.

## What it actually does

```bash
pip install -e .
sixth demo          # the whole loop, end to end, on synthetic data
```

One cycle, in full:

```
prereg → seal holdout → walk-forward → break → sealed OOS → post-mortem
       → verdict → record → harvest lessons → propose next
```

```bash
sixth add "A 20/100 MA cross captures trend persistence and beats buy-and-hold." --tags trend
sixth prereg a-20-100-ma-cross-captures-trend-persistence-and \
    --must "sharpe >= 0.8" --must "max_drawdown <= 0.25" \
    --rationale "trends persist long enough that a lagging filter still catches them"
sixth run a-20-100-ma-cross-captures-trend-persistence-and \
    --strategy sma_cross --grid fast=10,20,40 slow=60,100,200 --create-proposals
```

and out comes a verdict you did not get to choose after the fact:

```
VERDICT: REGIME_CONDITIONAL   (experiment #1)

WALK-FORWARD (development data)
  out-of-sample Sharpe 0.162
  Sharpe decay IS->OOS 0.183
  parameter stability  80%

BREAKER (adversarial)
  probes survived      56% (10/18)
  edge dies at         110 bps round-turn
  vs random null       p = 0.129

SEALED HOLDOUT (touched once, just now)
  Sharpe               0.567
  Deflated Sharpe      0.072 over 45 trials

WRITTEN TO GRAPH (10 lessons)
  [regime]         Loses money in down/calm: Sharpe -3.41 over 30 bars...
  [execution]      Edge disappears at roughly 110 bps round-turn cost...
  [concentration]  Removing the best regime (up/normal) leaves Sharpe -0.23...
  [null]           Does not beat exposure-matched random signals (p = 0.129)...

NEXT HYPOTHESES IMPLIED BY THIS FAILURE
  - ...but flat during down/calm, flat/calm, flat/normal   [created]
  - ...traded only during up/calm, up/normal, up/stressed  [created]
```

## The part that closes the loop

`sixth context` renders the whole graph as a block you paste above your next
request to Claude Code (or any coding agent):

```bash
sixth context --focus "mean reversion on intraday bars"
```

```markdown
## Prior research state (do not re-derive)

This graph holds 13 hypotheses across 3 recorded experiments and 105 total
trials. Treat everything below as already established.

### Works only in specific regimes
- **a-20-100-moving-average-cross...** — holds in: up/calm, up/normal
  - bleeds in: down/calm, flat/calm, flat/normal

### Recurring failure patterns in this research programme
- [overfit] Sharpe decays 0.54 from in-sample to out-of-sample across 5 folds.
- [execution] Edge disappears at roughly 22 bps round-turn cost.
- [concentration] Removing the best regime leaves Sharpe -0.10.

### Rules for the next proposal
1. It must not duplicate anything under Refuted.
2. State it as a falsifiable claim with numeric must-conditions.
3. Assume 105 trials have already been spent; the Deflated Sharpe bar rises
   with every additional one.
```

That block is the whole point. The agent that writes your next strategy starts
from everything the last forty attempts learned, instead of from zero.

## Five design commitments

**1. Preregistration, borrowed from clinical trials.** You write down what you
expect before the test runs. It is hashed and sealed, and the verdict is computed
by machine against that record. You cannot move the goalposts on a strategy that
preregistered its own thesis.

**2. Append-only, enforced by the database.** SQLite triggers refuse `UPDATE` and
`DELETE` on preregistrations and experiments. A preregistration you can edit
afterwards is not a preregistration.

```python
>>> graph.conn.execute("UPDATE experiments SET verdict='confirmed'")
sqlite3.IntegrityError: experiments are append-only: record a new run instead
```

**3. Every result is regime-tagged.** "This failed" is nearly worthless. "This
failed in `down/stressed` and made all its money in `up/calm`, which was 18% of
the sample" is a research programme. `sixth regime down/stressed` lists
everything known to bleed there — including strategies that pass overall.

**4. The graph counts your trials, so the Deflated Sharpe is honest.** A search
over 4,000 combinations that reports its best result has an `n_trials` of 4,000,
not 1. The graph is the only thing that watched every attempt, so it is the only
thing that can supply that number:

| trials spent | Deflated Sharpe |
|---|---|
| 45 | 0.072 |
| 90 | 0.024 |
| 105 | 0.003 |

Same strategy, same data. The bar rises as your programme spends its search
budget — which is what should happen, and never does when the count is thrown
away.

**5. The kill switch lives in code, never in a prompt.** A risk limit written
into a system prompt is a suggestion, and a sufficiently motivated reasoning
chain will argue its way past it. `RiskGate.check()` returns a smaller number, or
raises. It has no text interface, so there is no argument to make to it.

```python
gate = RiskGate(RiskLimits(max_position=0.5, max_drawdown=0.10),
                equity=100_000)
gate.check(Order("AAPL", 5.0))     # -> 0.25, clamped
gate.mark(equity=85_000)           # -15% from peak
gate.check(Order("AAPL", 0.1))     # -> KillSwitchTripped
```

## What's in the box

| module | what it does |
|---|---|
| `graph.py` | **the hypothesis graph** — the sixth part, and the reason this repo exists |
| `prereg.py` | sealed, hash-verified expectations and machine-computed verdicts |
| `stats.py` | Sharpe, PSR, **Deflated Sharpe**, stationary-bootstrap Monte Carlo |
| `backtest.py` | bar-by-bar engine with costs, slippage, borrow; structurally lookahead-free |
| `walkforward.py` | purged/embargoed walk-forward and the sealed holdout |
| `breaker.py` | the adversary: 2×/4× costs, worst historical windows, execution lag, parameter neighbourhoods, random-null tests |
| `postmortem.py` | the autopsy — where it bled, by regime, by episode, by cost drag |
| `lessons.py` | turns autopsies into typed graph rows and proposes the next hypotheses |
| `context.py` | renders the graph as agent context. **This is the loop closing.** |
| `regimes.py` | causal trend/volatility labelling |
| `live.py` | paper broker and the risk gate |

## Using it as a library

```python
from sixth import HypothesisGraph, Expectation, run_cycle, synthetic_bars, brief

graph = HypothesisGraph("research.sqlite")
h = graph.add("A 20/100 MA cross beats buy-and-hold risk-adjusted.", tags=["trend"])
graph.preregister(h.id, Expectation.parse(["sharpe >= 0.8", "max_drawdown <= 0.25"]))

result = run_cycle(graph, h.id, "sma_cross", synthetic_bars(),
                   grid={"fast": [10, 20, 40], "slow": [60, 100, 200]},
                   create_proposals=True)

print(result.verdict)          # 'regime_conditional'
print(brief(graph))            # the block to paste into your next agent prompt
```

Your own strategy is a function from bars to target weights in `[-1, 1]`:

```python
from sixth import register

@register("my_idea", "Long when today closed above the 50-bar high.", window=50)
def my_idea(bars, window=50):
    return [1.0 if i >= window and bars.close[i] >= max(bars.close[i-window:i])
            else 0.0 for i in range(len(bars))]
```

The engine computes weight `i` from data through bar `i` and fills it at bar
`i+1`'s open, so lookahead is structurally impossible rather than merely
discouraged. There is a test that proves it.

## Bring your own data

```bash
sixth run my-hypothesis --strategy sma_cross --csv ~/data/SPY.csv --symbol SPY
```

Any CSV with a `date` and `close` column; `open/high/low/volume` are used when
present. With no `--csv`, a deterministic regime-switching synthetic series is
generated, so the whole loop runs offline with no vendor account.

## Scope, and what this is not

- **It does not trade.** `live.py` ships a paper broker and a risk gate. It routes
  no real orders and holds no venue credentials. `BrokerAdapter` is the interface
  to implement if you go live; read the whole class first.
- **It is not investment advice**, and a passing verdict is not a
  recommendation. It is a research ledger that is harder to fool than a
  spreadsheet.
- **The synthetic generator is a fixture, not a market.** It exists so tests have
  known ground truth and so the demo runs anywhere. Point it at real bars before
  believing anything.
- **Backtests are not returns.** Everything here is built to make that gap
  visible — which is why the breaker, the trial counter and the sealed holdout
  exist.

## Tests

```bash
python -m pytest tests -q     # 153 tests, no dependencies beyond pytest
```

Including the ones that matter: that a strategy cannot trade the bar that
produced its own signal, that a lookahead strategy dies under a one-bar delay,
that a coin flip is never certified as an edge, that training windows never touch
their test windows, and that the database refuses to let you edit a sealed
expectation.

## Provenance

Built from the argument in
[this article](https://x.com/antpalkin/status/2085431604906766385) by
[@antpalkin](https://x.com/antpalkin), which lays out the six-part loop, notes
that five parts have collapsed into a chat box, and identifies the sixth —
folding results back in as a persistent world model — as the piece nobody has
assembled end to end. This is an attempt at that piece.

MIT.
