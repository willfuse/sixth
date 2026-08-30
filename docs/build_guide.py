"""Build the 'how to use sixth' guide."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

INK      = colors.HexColor("#16181d")
MUTED    = colors.HexColor("#5c6370")
RULE     = colors.HexColor("#d8dce3")
ACCENT   = colors.HexColor("#a3401f")
CODE_BG  = colors.HexColor("#f5f3f0")
CODE_INK = colors.HexColor("#2b2f38")
GOOD     = colors.HexColor("#2f6b46")

PAGE_W, PAGE_H = LETTER
M = 0.85 * inch

ss = getSampleStyleSheet()

def style(name, **kw):
    base = kw.pop("parent", ss["BodyText"])
    return ParagraphStyle(name, parent=base, **kw)

Title    = style("T", fontName="Helvetica-Bold", fontSize=30, leading=34,
                 textColor=INK, spaceAfter=6)
Sub      = style("S", fontName="Helvetica", fontSize=13, leading=18,
                 textColor=MUTED, spaceAfter=2)
H1       = style("H1", fontName="Helvetica-Bold", fontSize=15.5, leading=19,
                 textColor=INK, spaceBefore=17, spaceAfter=7)
H2       = style("H2", fontName="Helvetica-Bold", fontSize=11.2, leading=14.5,
                 textColor=ACCENT, spaceBefore=12, spaceAfter=5)
Body     = style("B", fontSize=9.9, leading=14.4, textColor=INK, spaceAfter=7,
                 alignment=TA_LEFT)
Small    = style("Sm", fontSize=8.7, leading=12.4, textColor=MUTED, spaceAfter=6)
Code     = style("C", fontName="Courier", fontSize=8.5, leading=12.2,
                 textColor=CODE_INK, spaceAfter=0, spaceBefore=0,
                 leftIndent=9, rightIndent=6)
Bullet   = style("Bu", parent=Body, leftIndent=15, bulletIndent=4, spaceAfter=4.5)
Lead     = style("L", fontSize=11.4, leading=16.5, textColor=INK, spaceAfter=9)


def code(lines, bg=CODE_BG):
    """A code block that survives page breaks as one unit."""
    txt = "<br/>".join(
        l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
         .replace(" ", "&nbsp;") or "&nbsp;" for l in lines)
    t = Table([[Paragraph(txt, Code)]], colWidths=[PAGE_W - 2 * M])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
    ]))
    return [t, Spacer(1, 9)]


def table(rows, widths, header=True, fs=8.8):
    # Plain strings do not wrap inside a Table cell, so every cell becomes a
    # Paragraph. Without this, long cells overflow into the next column.
    cell = style("tc", fontSize=fs, leading=fs * 1.42, textColor=INK, spaceAfter=0)
    head = style("th", fontName="Helvetica-Bold", fontSize=fs, leading=fs * 1.42,
                 textColor=colors.white, spaceAfter=0)
    wrapped = [[Paragraph(c, head if (header and r == 0) else cell)
                if isinstance(c, str) else c for c in row]
               for r, row in enumerate(rows)]
    t = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0)
    st = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
    ]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), INK),
               ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
               ("TOPPADDING", (0, 0), (-1, 0), 6)]
    t.setStyle(TableStyle(st))
    return [t, Spacer(1, 10)]


def callout(title, body, tint="#f3f6f3", edge=GOOD):
    inner = [Paragraph(f"<b>{title}</b>", style("ct", fontSize=9.4, leading=13,
                                                textColor=edge, spaceAfter=3)),
             Paragraph(body, style("cb", fontSize=9.2, leading=13.3, textColor=INK))]
    t = Table([[inner]], colWidths=[PAGE_W - 2 * M])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(tint)),
        ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, edge),
    ]))
    return [t, Spacer(1, 10)]


def bullets(items):
    return [Paragraph(f"<font color='#a3401f'>•</font>&nbsp;&nbsp;{i}",
                      Bullet) for i in items]


def rule(space_before=3, space_after=9):
    t = Table([[""]], colWidths=[PAGE_W - 2 * M], rowHeights=[0.6])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), RULE)]))
    return [Spacer(1, space_before), t, Spacer(1, space_after)]


# ---------------------------------------------------------------- page chrome
def chrome(canv, doc):
    canv.saveState()
    if doc.page > 1:
        canv.setFont("Helvetica", 7.6)
        canv.setFillColor(MUTED)
        canv.drawString(M, PAGE_H - M + 22, "sixth — how to use it")
        canv.drawRightString(PAGE_W - M, PAGE_H - M + 22, "github.com/willfuse/sixth")
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.5)
        canv.line(M, PAGE_H - M + 16, PAGE_W - M, PAGE_H - M + 16)
        canv.drawCentredString(PAGE_W / 2, M - 26, str(doc.page))
    canv.restoreState()


def cover_chrome(canv, doc):
    canv.saveState()
    canv.setFillColor(ACCENT)
    canv.rect(0, PAGE_H - 13, PAGE_W, 13, stroke=0, fill=1)
    canv.setFont("Helvetica", 7.6)
    canv.setFillColor(MUTED)
    canv.drawCentredString(PAGE_W / 2, M - 26, "1")
    canv.restoreState()


doc = BaseDocTemplate("sixth-how-to-use-it.pdf", pagesize=LETTER,
                      leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                      title="sixth - how to use it",
                      author="willfuse", subject="Usage guide for the sixth package")
frame = Frame(M, M, PAGE_W - 2 * M, PAGE_H - 2 * M, id="f",
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
doc.addPageTemplates([
    PageTemplate(id="cover", frames=[frame], onPage=cover_chrome),
    PageTemplate(id="body", frames=[frame], onPage=chrome),
])

S = []


def A(*items):
    for i in items:
        S.extend(i) if isinstance(i, list) else S.append(i)

# ============================================================== COVER
A(Spacer(1, 46))
A(Paragraph("sixth", Title))
A(Paragraph("The part of the quant research loop that remembers.", Sub))
A(Paragraph("How to use it", style("cv", fontName="Helvetica-Bold", fontSize=13,
                                   leading=17, textColor=ACCENT, spaceBefore=13)))
A(Spacer(1, 16))
A(*rule(0, 14))

A(Paragraph(
    "A hedge fund is one cycle repeating: <b>research → code → backtest → "
    "live → post-mortem → fine-tune</b>. Five of those six work today — you can "
    "describe a strategy in English and have an agent write, test and deploy it in "
    "about ninety seconds.", Lead))
A(Paragraph(
    "The sixth doesn't work, and it isn't hard. It's just missing. The lesson from a "
    "losing test lands in a log file and stays there; the next strategy your agent "
    "writes has never heard of it. Every cycle restarts from the same place, so nothing "
    "compounds.", Lead))
A(Paragraph(
    "<b>sixth</b> is that missing part: a persistent, queryable graph of what you tested, "
    "what failed, <i>in which market regime</i>, and what you have never tried at all.", Lead))

A(Spacer(1, 10))
A(*callout(
    "Sixty seconds from here to a full research cycle",
    "<font face='Courier' size='9'>git clone https://github.com/willfuse/sixth &amp;&amp; "
    "cd sixth<br/>pip install -e .<br/>sixth demo</font><br/><br/>"
    "No API key, no data vendor, no network. Zero dependencies — pure Python "
    "standard library, 3.9 and up.", tint="#f6f2ef", edge=ACCENT))

A(Spacer(1, 6))
A(*table([
    ["What you run", "What you get back"],
    ["sixth add", "a hypothesis stated, checked against near-duplicates you already tested"],
    ["sixth prereg", "your expectation hashed and sealed, before any result exists"],
    ["sixth run", "walk-forward, an adversarial breaker pass, a sealed-holdout verdict"],
    ["sixth context", "everything learned, formatted to paste into your next agent prompt"],
], [1.45 * inch, PAGE_W - 2 * M - 1.45 * inch]))

A(Paragraph(
    "Built from the argument in antpalkin's article on the self-improving trading "
    "machine, which identifies the fine-tune step as the piece nobody has assembled "
    "end to end. Research tooling only — it places no orders and is not investment "
    "advice. See section 8.", Small))

A(NextPageTemplate("body"))
A(PageBreak())

# ============================================================== 1. INSTALL
A(Paragraph("1 &nbsp;Install and first run", H1))
A(Paragraph(
    "You need Python 3.9 or newer. Nothing else — there is no numpy, no pandas, no "
    "vendor SDK. The whole package is standard library, so it installs anywhere and "
    "runs offline.", Body))
A(*code([
    "git clone https://github.com/willfuse/sixth",
    "cd sixth",
    "python3 -m venv .venv && source .venv/bin/activate",
    "pip install -e .",
    "",
    "sixth demo          # the entire loop, end to end, on synthetic data",
]))
A(Paragraph(
    "<b>sixth demo</b> states three hypotheses, seals an expectation for each, runs a "
    "full cycle on all three, and prints the research context that accumulated. Read its "
    "output once before anything else — it is the whole argument of the package in "
    "about ninety lines.", Body))
A(Paragraph(
    "Every command takes <font face='Courier' size='8.5'>--db</font> to pick a graph "
    "file. It is a top-level flag, so it goes <i>before</i> the subcommand:", Body))
A(*code([
    "sixth --db research.sqlite status        # correct",
    "sixth status --db research.sqlite        # error: unrecognized arguments",
    "",
    "export SIXTH_DB=~/research.sqlite        # or set it once",
]))
A(Paragraph(
    "The default is <font face='Courier' size='8.5'>sixth.sqlite</font> in the current "
    "directory. One file holds everything: hypotheses, sealed expectations, every result, "
    "every lesson. Back it up like source code — it is the asset.", Body))

# ============================================================== 2. WORKFLOW
A(Paragraph("2 &nbsp;The workflow, in four commands", H1))
A(Paragraph(
    "A research cycle is always these four steps in this order. The order is enforced: "
    "you cannot run a test that has no sealed expectation.", Body))

A(Paragraph("Step 1 — State the hypothesis", H2))
A(*code([
    'sixth add "A 20/100 moving-average cross captures trend persistence \\',
    '           and beats buy-and-hold risk-adjusted." --tags trend,equity',
]))
A(Paragraph(
    "This runs nothing. Stating an idea and testing it are separate acts, which is why "
    "<b>sixth frontier</b> can later tell you what you thought of but never got to. If the "
    "statement closely matches something already in the graph, the command refuses and "
    "shows you the earlier one — override with "
    "<font face='Courier' size='8.5'>--force</font> if it really is different.", Body))

A(Paragraph("Step 2 — Seal what you expect, before you look", H2))
A(*code([
    'sixth prereg a-20-100-moving-average-cross-captures-trend-persistence-and \\',
    '    --must "sharpe >= 0.8" \\',
    '    --must "max_drawdown <= 0.25" \\',
    '    --rationale "trends persist long enough that a lagging filter still catches them"',
]))
A(Paragraph(
    "Borrowed from clinical trials. The conditions are hashed, timestamped and written to "
    "the database, which then refuses to update or delete the row. The verdict is computed "
    "by machine against this record, so you cannot move the goalposts on a strategy that "
    "preregistered its own thesis.", Body))
A(Paragraph(
    "<b>--must</b> conditions decide confirmed versus refuted. <b>--should</b> conditions "
    "are secondary: failing one alone downgrades the result to <i>inconclusive</i> rather "
    "than refuting it. At least one <b>--must</b> is required, because an expectation with "
    "nothing falsifiable in it is not an expectation.", Body))

A(Paragraph("Step 3 — Run the cycle", H2))
A(*code([
    'sixth run a-20-100-moving-average-cross-captures-trend-persistence-and \\',
    '    --strategy sma_cross \\',
    '    --grid fast=10,20,40 slow=60,100,200 \\',
    '    --create-proposals',
]))
A(Paragraph("One command does all of this, in order:", Body))
A(*bullets([
    "<b>Seals a holdout.</b> The last 20% of the data is split off and physically not "
    "handed to anything that fits parameters.",
    "<b>Walks forward</b> across the rest with a purge and embargo between each training "
    "and test window, so a 100-bar indicator can never train on its own test set.",
    "<b>Counts every trial.</b> Nine parameter combinations over five folds is 45 trials, "
    "not one.",
    "<b>Runs the breaker</b> — an adversary that only tries to kill the strategy.",
    "<b>Looks at the sealed holdout once</b>, with the parameters now frozen.",
    "<b>Writes the autopsy, the verdict and the lessons</b> into the graph, and proposes "
    "the follow-up hypotheses the failure implies.",
]))

A(Paragraph("Step 4 — Hand it all to your next agent", H2))
A(*code(['sixth context --focus "mean reversion on intraday bars"']))
A(Paragraph(
    "This is the step that closes the loop, and the reason the other three are worth "
    "doing. Page 6 covers it.", Body))

A(PageBreak())

# ============================================================== 3. READING OUTPUT
A(Paragraph("3 &nbsp;Reading what comes back", H1))
A(Paragraph(
    "A cycle prints four blocks. Here is a real one, with what each number is telling "
    "you.", Body))

A(*code([
    "VERDICT: REGIME_CONDITIONAL   (experiment #1)",
    "",
    "WALK-FORWARD (development data)",
    "  out-of-sample Sharpe 0.162",
    "  Sharpe decay IS->OOS 0.183",
    "  parameter stability  80%",
    "  folds profitable     80%",
    "",
    "BREAKER (adversarial)",
    "  probes survived      56% (10/18)",
    "  edge dies at         110 bps round-turn",
    "  vs random null       p = 0.129",
    "",
    "SEALED HOLDOUT (touched once, just now)",
    "  Sharpe               0.567",
    "  max drawdown         21.9%",
    "  Deflated Sharpe      0.072 over 45 trials",
]))

A(*table([
    ["Line", "What it means", "Worry when"],
    ["out-of-sample Sharpe", "performance on windows never used to fit parameters",
     "below your must-condition"],
    ["Sharpe decay", "in-sample minus out-of-sample, averaged over folds",
     "above ~0.5: fitting noise"],
    ["parameter stability", "share of folds that chose the same parameters",
     "below 50%: nothing stable to deploy"],
    ["probes survived", "share of adversarial probes the strategy lived through",
     "any kill you can't explain"],
    ["edge dies at", "round-turn cost that zeroes the edge",
     "near your real execution cost"],
    ["vs random null", "p-value against exposure-matched coin flips",
     "above 0.05: indistinguishable from chance"],
    ["Deflated Sharpe", "probability the edge survives the search that found it",
     "below ~0.5: the search produced it"],
], [1.24 * inch, 2.72 * inch, PAGE_W - 2 * M - 3.96 * inch], fs=8.4))

A(Paragraph("The five verdicts", H2))
A(*table([
    ["Verdict", "Meaning"],
    ["confirmed", "every must-condition held, and every should-condition too"],
    ["regime_conditional", "failed overall, but held completely inside at least one regime — "
     "usually the most useful outcome you can get"],
    ["refuted", "failed its must-conditions everywhere, in every regime"],
    ["inconclusive", "must-conditions held but a should-condition failed, or nothing "
     "falsifiable was stated"],
    ["retired", "you set it aside deliberately; kept so it is never re-tried by accident"],
], [1.24 * inch, PAGE_W - 2 * M - 1.24 * inch]))

A(*callout(
    "Why a refuted result is worth as much as a confirmed one",
    "Both cost the same to produce, and only one of them is usually kept. A refutation "
    "with a regime attached tells your next agent where not to spend its next forty "
    "attempts. That is the asset this package exists to stop you throwing away.",
    tint="#f6f2ef", edge=ACCENT))

# ============================================================== 4. WHY DSR
A(Paragraph("4 &nbsp;The number the graph makes honest", H1))
A(Paragraph(
    "A Sharpe ratio does not tell you whether an edge is real. It tells you what the best "
    "of however many things you tried happened to score. Search four thousand parameter "
    "combinations against noise and one of them will look excellent.", Body))
A(Paragraph(
    "The Deflated Sharpe Ratio corrects for exactly this — but it needs an input almost "
    "nobody keeps: <b>how many variants you actually looked at</b>. The graph watched every "
    "one, so it can supply the real number instead of a flattering one.", Body))

A(*table([
    ["Trials the graph has watched", "Deflated Sharpe", "Reading"],
    ["1 (what you would report)", "0.930", "looks like a strong edge"],
    ["45 (this search alone)", "0.072", "probably the search, not the market"],
    ["90 (after the next cycle)", "0.024", "the bar has risen"],
    ["105 (after the one after that)", "0.003", "nothing here survives the count"],
], [2.3 * inch, 1.25 * inch, PAGE_W - 2 * M - 3.55 * inch]))

A(Paragraph(
    "Same strategy, same data, same returns. Only the honest trial count changes. This is "
    "what a research programme with a memory looks like from the inside: the standard of "
    "proof rises as you spend your search budget, which is what should happen and never "
    "does when the count is discarded after each run.", Body))

# ============================================================== 5. QUERY
A(Paragraph("5 &nbsp;Asking the graph questions", H1))
A(Paragraph(
    "Once a few cycles have run, the graph answers things no log file can.", Body))

A(*table([
    ["Command", "Question it answers"],
    ["sixth status", "how big is this programme, and how many trials has it spent"],
    ["sixth list --status refuted", "what have I already ruled out"],
    ["sixth frontier", "what did I think of and never actually test"],
    ["sixth regime down/stressed", "what is known to bleed in a falling, volatile market — "
     "including strategies that pass overall"],
    ["sixth show &lt;slug&gt;", "everything about one idea: every run, every regime, every "
     "lesson, what it was derived from"],
    ["sixth lessons --kind overfit", "every time this programme has fooled itself the same way"],
    ["sixth export", "the whole graph as JSON, for diffing or feeding to a model"],
], [1.98 * inch, PAGE_W - 2 * M - 1.98 * inch]))

A(Paragraph(
    "The regime query is the one worth learning first. Results are stored sliced by market "
    "condition — trend direction crossed with volatility level, both computed causally "
    "from trailing windows only. So \"this failed\" becomes \"this failed in "
    "<font face='Courier' size='8.5'>down/stressed</font> and made all of its money in "
    "<font face='Courier' size='8.5'>up/calm</font>, which was 18% of the sample\", which "
    "is an actionable research direction rather than a dead end.", Body))

A(*code([
    "$ sixth regime down/normal",
    "",
    "2 hypotheses lose money in down/normal",
    "",
    "  a-20-100-moving-average-cross...   sharpe -2.82 over 355 bars",
    "    A 20/100 moving-average cross captures trend persistence...",
    "  simple-60-day-time-series-momentum...  sharpe -3.61 over 84 bars",
    "    Simple 60-day time-series momentum earns a positive Sharpe...",
]))

# ============================================================== 6. CONTEXT
A(Paragraph("6 &nbsp;Closing the loop", H1))
A(Paragraph(
    "Everything so far is bookkeeping. This is the part that makes it compound.", Body))
A(Paragraph(
    "<b>sixth context</b> renders the entire graph as a markdown block designed to sit at "
    "the top of a prompt. Paste it above your next request to Claude Code, or pipe it into "
    "whatever writes your strategies:", Body))
A(*code([
    "sixth context --focus \"intraday mean reversion\" | pbcopy",
    "sixth context --json | jq '.frontier'      # for programmatic use",
]))
A(Paragraph("What it emits:", Body))
A(*code([
    "## Prior research state (do not re-derive)",
    "",
    "This graph holds 13 hypotheses across 3 recorded experiments and 105",
    "total trials. Treat everything below as already established.",
    "",
    "### Works only in specific regimes",
    "- a-20-100-moving-average-cross... ",
    "  - holds in: up/calm, up/normal, up/stressed",
    "  - bleeds in: down/calm, flat/calm, flat/normal",
    "",
    "### Recurring failure patterns in this research programme",
    "- [overfit] Sharpe decays 0.54 from in-sample to out-of-sample.",
    "- [execution] Edge disappears at roughly 22 bps round-turn cost.",
    "- [concentration] Removing the best regime leaves Sharpe -0.10.",
    "",
    "### Frontier - stated but never tested",
    "- ...traded only during up/calm, up/normal, up/stressed",
    "",
    "### Rules for the next proposal",
    "1. It must not duplicate anything under Refuted.",
    "2. State it as a falsifiable claim with numeric must-conditions.",
    "3. Assume 105 trials have already been spent; the Deflated Sharpe",
    "   bar rises with every additional one.",
], bg=colors.HexColor("#f2f4f7")))
A(Paragraph(
    "The agent writing your next strategy now starts from everything the last forty "
    "attempts learned, instead of from zero. It knows which ideas are spent, which regimes "
    "eat this style of signal, and how many trials the programme has already burned. That "
    "is the difference between a trading bot and something that compounds.", Body))
A(Paragraph(
    "<font face='Courier' size='8.5'>examples/agent_loop.py</font> in the repo shows the "
    "full pattern, including the near-duplicate check that stops an agent proposing "
    "something the graph has already settled.", Body))

A(PageBreak())

# ============================================================== 7. YOUR DATA / STRATEGY
A(Paragraph("7 &nbsp;Your own data and your own strategies", H1))

A(Paragraph("Real bars", H2))
A(*code([
    "sixth run my-hypothesis --strategy sma_cross \\",
    "    --csv ~/data/SPY.csv --symbol SPY \\",
    "    --commission-bps 1 --slippage-bps 3",
]))
A(Paragraph(
    "Any CSV with <font face='Courier' size='8.5'>date</font> and "
    "<font face='Courier' size='8.5'>close</font> columns works; "
    "<font face='Courier' size='8.5'>open</font>, "
    "<font face='Courier' size='8.5'>high</font>, "
    "<font face='Courier' size='8.5'>low</font> and "
    "<font face='Courier' size='8.5'>volume</font> are used when present. With no "
    "<font face='Courier' size='8.5'>--csv</font>, a deterministic regime-switching "
    "synthetic series is generated so everything runs offline. That series is a "
    "<i>test fixture with known ground truth</i>, not a market — point it at real bars "
    "before believing any number.", Body))
A(Paragraph(
    "Every stored result carries a fingerprint of the data it ran on, so a result can "
    "never be silently compared against a different dataset.", Body))

A(Paragraph("Writing a strategy", H2))
A(Paragraph(
    "A strategy is a function from bars to target weights in "
    "<font face='Courier' size='8.5'>[-1, 1]</font>. That is the entire contract:", Body))
A(*code([
    "from sixth import register",
    "",
    '@register("breakout", "Long when today closed at a 50-bar high.", window=50)',
    "def breakout(bars, window=50):",
    "    return [1.0 if i >= window and bars.close[i] >= max(bars.close[i-window:i])",
    "            else 0.0",
    "            for i in range(len(bars))]",
]))
A(Paragraph(
    "The engine computes weight <font face='Courier' size='8.5'>i</font> from data through "
    "bar <font face='Courier' size='8.5'>i</font> and fills it at bar "
    "<font face='Courier' size='8.5'>i+1</font>'s open. Lookahead is structurally "
    "impossible rather than merely discouraged — a strategy literally cannot trade the "
    "bar that produced its own signal. There is a test that proves it, and another that "
    "confirms a peeking strategy dies the moment a one-bar delay is applied.", Body))
A(Paragraph(
    "Run <b>sixth strategies</b> for the built-ins: "
    "<font face='Courier' size='8.5'>sma_cross</font>, "
    "<font face='Courier' size='8.5'>sma_cross_ls</font>, "
    "<font face='Courier' size='8.5'>momentum</font>, "
    "<font face='Courier' size='8.5'>mean_reversion</font>, "
    "<font face='Courier' size='8.5'>buy_hold</font>, "
    "<font face='Courier' size='8.5'>flat</font> and "
    "<font face='Courier' size='8.5'>random_signal</font>. The last three are null models "
    "— the things your idea has to beat.", Body))

A(Paragraph("As a library", H2))
A(*code([
    "from sixth import (HypothesisGraph, Expectation, run_cycle,",
    "                   synthetic_bars, brief)",
    "",
    'graph = HypothesisGraph("research.sqlite")',
    'h = graph.add("A 20/100 MA cross beats buy-and-hold.", tags=["trend"])',
    'graph.preregister(h.id, Expectation.parse(["sharpe >= 0.8",',
    '                                           "max_drawdown <= 0.25"]))',
    "",
    'result = run_cycle(graph, h.id, "sma_cross", synthetic_bars(),',
    '                   grid={"fast": [10, 20, 40], "slow": [60, 100, 200]},',
    "                   create_proposals=True)",
    "",
    "print(result.verdict)      # 'regime_conditional'",
    "print(brief(graph))        # the block for your next agent prompt",
]))

A(PageBreak())

# ============================================================== 8. RISK / SCOPE
A(Paragraph("8 &nbsp;The risk gate, and what this will not do", H1))

A(Paragraph("The kill switch lives in code", H2))
A(Paragraph(
    "A risk limit written into a system prompt is a suggestion, and a sufficiently "
    "motivated reasoning chain will argue its way past it. So "
    "<font face='Courier' size='8.5'>RiskGate</font> has no text interface at all. It "
    "returns a smaller number, or it raises. There is no argument to make to it.", Body))
A(*code([
    "from sixth import RiskGate, RiskLimits",
    "from sixth.live import Order",
    "",
    "gate = RiskGate(RiskLimits(max_position=0.5, max_drawdown=0.10,",
    "                           max_daily_loss=0.02, allow_shorts=False),",
    "                equity=100_000)   # the gate needs a starting equity",
    "",
    'gate.check(Order("AAPL", 5.0))     # -> 0.25   clamped, not refused',
    'gate.check(Order("AAPL", -1.0))    # -> 0.0    shorts disabled',
    "gate.mark(equity=85_000)           # -15% from peak",
    'gate.check(Order("AAPL", 0.1))     # -> KillSwitchTripped',
]))
A(Paragraph(
    "Once tripped it stays tripped — a later recovery in equity does not resume trading. "
    "Every clamp is journalled with the rule that fired, which is what the post-mortem "
    "reads later.", Body))

A(Paragraph("Scope", H2))
A(*bullets([
    "<b>It does not trade.</b> The package ships a paper broker and the risk gate. It "
    "routes no real orders and holds no venue credentials. "
    "<font face='Courier' size='8.5'>BrokerAdapter</font> is the interface to implement "
    "if you go live; read the whole class first, and route every order through the gate.",
    "<b>It is not investment advice.</b> A <i>confirmed</i> verdict is not a "
    "recommendation. It means a claim you wrote down in advance survived a set of "
    "mechanical checks on historical data.",
    "<b>Backtests are not returns.</b> Everything in the package — the breaker, the "
    "trial counter, the sealed holdout, the random-null test — exists to make that gap "
    "visible rather than to close it.",
    "<b>The synthetic market is a fixture.</b> It gives the tests known ground truth and "
    "lets the demo run anywhere. It is not a claim about real markets.",
]))

A(Paragraph("9 &nbsp;Where things live", H1))
A(*table([
    ["Path", "What it is"],
    ["src/sixth/graph.py", "the hypothesis graph — the sixth part, and the reason the repo exists"],
    ["src/sixth/prereg.py", "sealed expectations and machine-computed verdicts"],
    ["src/sixth/stats.py", "Sharpe, PSR, Deflated Sharpe, stationary-bootstrap Monte Carlo"],
    ["src/sixth/backtest.py", "the engine: costs, slippage, borrow, no lookahead"],
    ["src/sixth/walkforward.py", "purged and embargoed folds, plus the sealed holdout"],
    ["src/sixth/breaker.py", "the adversary that only tries to kill strategies"],
    ["src/sixth/postmortem.py", "the autopsy: where it bled, by regime and by episode"],
    ["src/sixth/lessons.py", "autopsies → graph rows → proposed next hypotheses"],
    ["src/sixth/context.py", "renders the graph as agent context — the loop closing"],
    ["src/sixth/live.py", "paper broker and the risk gate"],
    ["examples/agent_loop.py", "driving the cycle from an agent instead of the CLI"],
    ["tests/", "153 tests, no dependencies beyond pytest"],
], [1.72 * inch, PAGE_W - 2 * M - 1.72 * inch], fs=8.5))

A(Paragraph(
    "<font face='Courier' size='8.5'>python -m pytest tests -q</font> — including the "
    "ones that matter: that a strategy cannot trade the bar that produced its own signal, "
    "that a lookahead strategy dies under a one-bar delay, that a coin flip is never "
    "certified as an edge, that training windows never touch their test windows, and that "
    "the database refuses to let you edit a sealed expectation.", Body))

A(*rule(6, 8))
A(Paragraph(
    "github.com/willfuse/sixth &nbsp;·&nbsp; MIT &nbsp;·&nbsp; "
    "Built from the six-part loop described by @antpalkin. Research tooling; not "
    "investment advice.", Small))

# A heading stranded at the foot of a page is the one layout bug worth fixing
# globally: bind every heading to the block that follows it.
def bind_headings(flow):
    out, i = [], 0
    while i < len(flow):
        f = flow[i]
        is_head = isinstance(f, Paragraph) and f.style.name in ("H1", "H2")
        if is_head and i + 1 < len(flow):
            group = [f, flow[i + 1]]
            # Carry the trailing spacer along so the group is not split from it.
            if i + 2 < len(flow) and isinstance(flow[i + 2], Spacer):
                group.append(flow[i + 2])
            out.append(KeepTogether(group))
            i += len(group)
        else:
            out.append(f)
            i += 1
    return out


doc.build(bind_headings(S))
print("built")
