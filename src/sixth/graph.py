"""The hypothesis graph -- the sixth part of the loop.

Five parts of the quant cycle (research, code, backtest, live, post-mortem) all
work today. The lesson from a losing test still lands in a log file and dies
there, so the next strategy an agent writes starts from the same place as the
last one. This module is the fix: a persistent, queryable record of what was
tested, what failed, in which regime, and what has never been tried at all.

Design commitments
------------------
1. Negative results are first-class. `refuted` is a state you can query, rank and
   feed back into a prompt -- not an absence of a row.
2. Every result is regime-tagged. "Failed" without "where" is close to useless.
3. Preregistrations and experiments are append-only, enforced by SQLite triggers,
   not by convention. You cannot quietly rewrite a past expectation.
4. The graph counts your trials. That number is the input the Deflated Sharpe
   needs and the one nobody keeps honestly, because keeping it requires exactly
   this kind of store.

Storage is a single SQLite file. No server, no daemon, no vendor.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .prereg import CONFIRMED, INCONCLUSIVE, REFUTED, REGIME_CONDITIONAL, Expectation

STATUSES = (
    "untested",            # stated, never run
    CONFIRMED,             # held up against its own sealed expectation
    REFUTED,               # failed its must-conditions everywhere
    REGIME_CONDITIONAL,    # held only in some regimes -- the interesting one
    INCONCLUSIVE,          # ran, didn't settle it
    "retired",             # deliberately set aside; kept so it is not re-tried
)

EDGE_KINDS = ("refines", "contradicts", "derived_from", "generalizes", "duplicate_of")

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id          INTEGER PRIMARY KEY,
    slug        TEXT UNIQUE NOT NULL,
    statement   TEXT NOT NULL,
    spec_json   TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'untested',
    tags        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY,
    src_id      INTEGER NOT NULL REFERENCES hypotheses(id),
    dst_id      INTEGER NOT NULL REFERENCES hypotheses(id),
    kind        TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    UNIQUE(src_id, dst_id, kind)
);

CREATE TABLE IF NOT EXISTS preregs (
    id              INTEGER PRIMARY KEY,
    hypothesis_id   INTEGER NOT NULL REFERENCES hypotheses(id),
    expectation_json TEXT NOT NULL,
    sealed_at       TEXT NOT NULL,
    seal_hash       TEXT NOT NULL,
    consumed        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS experiments (
    id               INTEGER PRIMARY KEY,
    hypothesis_id    INTEGER NOT NULL REFERENCES hypotheses(id),
    prereg_id        INTEGER REFERENCES preregs(id),
    verdict          TEXT NOT NULL,
    config_json      TEXT NOT NULL DEFAULT '{}',
    metrics_json     TEXT NOT NULL DEFAULT '{}',
    regime_json      TEXT NOT NULL DEFAULT '{}',
    checks_json      TEXT NOT NULL DEFAULT '[]',
    data_fingerprint TEXT NOT NULL DEFAULT '',
    code_hash        TEXT NOT NULL DEFAULT '',
    n_trials         INTEGER NOT NULL DEFAULT 1,
    notes            TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lessons (
    id            INTEGER PRIMARY KEY,
    hypothesis_id INTEGER REFERENCES hypotheses(id),
    experiment_id INTEGER REFERENCES experiments(id),
    kind          TEXT NOT NULL,
    regime        TEXT NOT NULL DEFAULT '',
    text          TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    UNIQUE(hypothesis_id, kind, regime, text)
);

CREATE INDEX IF NOT EXISTS ix_exp_hyp ON experiments(hypothesis_id);
CREATE INDEX IF NOT EXISTS ix_les_hyp ON lessons(hypothesis_id);
CREATE INDEX IF NOT EXISTS ix_hyp_status ON hypotheses(status);

-- Append-only, enforced by the database. A preregistration you can edit after
-- the fact is not a preregistration.
CREATE TRIGGER IF NOT EXISTS preregs_no_update BEFORE UPDATE OF
    expectation_json, sealed_at, seal_hash, hypothesis_id ON preregs
BEGIN SELECT RAISE(ABORT, 'preregs are sealed: expectations cannot be edited'); END;

CREATE TRIGGER IF NOT EXISTS preregs_no_delete BEFORE DELETE ON preregs
BEGIN SELECT RAISE(ABORT, 'preregs are sealed: expectations cannot be deleted'); END;

CREATE TRIGGER IF NOT EXISTS experiments_no_update BEFORE UPDATE ON experiments
BEGIN SELECT RAISE(ABORT, 'experiments are append-only: record a new run instead'); END;

CREATE TRIGGER IF NOT EXISTS experiments_no_delete BEFORE DELETE ON experiments
BEGIN SELECT RAISE(ABORT, 'experiments are append-only: results cannot be deleted'); END;
"""


@dataclass
class Hypothesis:
    id: int
    slug: str
    statement: str
    spec: Dict[str, Any]
    status: str
    tags: List[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "slug": self.slug, "statement": self.statement,
                "spec": self.spec, "status": self.status, "tags": self.tags,
                "created_at": self.created_at, "updated_at": self.updated_at}


@dataclass
class Experiment:
    id: int
    hypothesis_id: int
    verdict: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    regimes: Dict[str, Dict[str, float]]
    checks: List[Dict[str, Any]]
    data_fingerprint: str
    code_hash: str
    n_trials: int
    notes: str
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "hypothesis"


class HypothesisGraph:
    """The persistent world model. Open it, write to it, never lose a result."""

    def __init__(self, path: str = "sixth.sqlite"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "HypothesisGraph":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # hypotheses
    # ------------------------------------------------------------------
    def add(self, statement: str, slug: Optional[str] = None,
            spec: Optional[Dict[str, Any]] = None,
            tags: Optional[Sequence[str]] = None,
            parent: Optional[str] = None,
            edge_kind: str = "derived_from") -> Hypothesis:
        """State a hypothesis. Does not run anything -- that is the point; the
        frontier of untested ideas is as much a part of the map as the failures."""
        slug = slug or self._unique_slug(slugify(statement))
        ts = _now()
        cur = self.conn.execute(
            "INSERT INTO hypotheses (slug, statement, spec_json, status, tags,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (slug, statement, json.dumps(spec or {}), "untested",
             ",".join(tags or []), ts, ts))
        self.conn.commit()
        h = self.get(cur.lastrowid)
        if parent:
            # Direction is child -> parent, so ancestry() can walk backwards to
            # the root idea.
            self.link(slug, parent, edge_kind)
        return h

    def _unique_slug(self, base: str) -> str:
        slug, i = base, 2
        while self.conn.execute("SELECT 1 FROM hypotheses WHERE slug=?", (slug,)).fetchone():
            slug, i = f"{base}-{i}", i + 1
        return slug

    def get(self, ref: Any) -> Hypothesis:
        row = self._row(ref)
        if row is None:
            raise KeyError(f"no hypothesis {ref!r}")
        return self._hyp(row)

    def _row(self, ref: Any) -> Optional[sqlite3.Row]:
        if isinstance(ref, Hypothesis):
            ref = ref.id
        col = "id" if isinstance(ref, int) else "slug"
        return self.conn.execute(
            f"SELECT * FROM hypotheses WHERE {col}=?", (ref,)).fetchone()

    def exists(self, ref: Any) -> bool:
        return self._row(ref) is not None

    @staticmethod
    def _hyp(row: sqlite3.Row) -> Hypothesis:
        return Hypothesis(
            id=row["id"], slug=row["slug"], statement=row["statement"],
            spec=json.loads(row["spec_json"]), status=row["status"],
            tags=[t for t in row["tags"].split(",") if t],
            created_at=row["created_at"], updated_at=row["updated_at"])

    def all(self, status: Optional[str] = None,
            tag: Optional[str] = None) -> List[Hypothesis]:
        q, args = "SELECT * FROM hypotheses", []
        where = []
        if status:
            where.append("status=?")
            args.append(status)
        if tag:
            where.append("(','||tags||',') LIKE ?")
            args.append(f"%,{tag},%")
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY id"
        return [self._hyp(r) for r in self.conn.execute(q, args)]

    def set_status(self, ref: Any, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        h = self.get(ref)
        self.conn.execute("UPDATE hypotheses SET status=?, updated_at=? WHERE id=?",
                          (status, _now(), h.id))
        self.conn.commit()

    def retag(self, ref: Any, tags: Sequence[str]) -> None:
        h = self.get(ref)
        self.conn.execute("UPDATE hypotheses SET tags=?, updated_at=? WHERE id=?",
                          (",".join(tags), _now(), h.id))
        self.conn.commit()

    # ------------------------------------------------------------------
    # edges
    # ------------------------------------------------------------------
    def link(self, src: Any, dst: Any, kind: str = "refines", note: str = "") -> None:
        if kind not in EDGE_KINDS:
            raise ValueError(f"edge kind must be one of {EDGE_KINDS}")
        a, b = self.get(src).id, self.get(dst).id
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (src_id, dst_id, kind, note, created_at)"
            " VALUES (?,?,?,?,?)", (a, b, kind, note, _now()))
        self.conn.commit()

    def neighbors(self, ref: Any, kind: Optional[str] = None
                  ) -> List[Tuple[str, str, Hypothesis]]:
        """Returns (direction, kind, hypothesis) for everything one hop away."""
        h = self.get(ref)
        out = []
        q = "SELECT * FROM edges WHERE src_id=?" + (" AND kind=?" if kind else "")
        for e in self.conn.execute(q, (h.id, kind) if kind else (h.id,)):
            out.append(("out", e["kind"], self.get(e["dst_id"])))
        q = "SELECT * FROM edges WHERE dst_id=?" + (" AND kind=?" if kind else "")
        for e in self.conn.execute(q, (h.id, kind) if kind else (h.id,)):
            out.append(("in", e["kind"], self.get(e["src_id"])))
        return out

    def ancestry(self, ref: Any, max_depth: int = 10) -> List[Hypothesis]:
        """Walk `derived_from`/`refines` edges back to the root idea."""
        chain, seen = [], set()
        cur = self.get(ref)
        for _ in range(max_depth):
            row = self.conn.execute(
                "SELECT dst_id FROM edges WHERE src_id=? AND kind IN"
                " ('derived_from','refines') ORDER BY id LIMIT 1", (cur.id,)).fetchone()
            if not row or row["dst_id"] in seen:
                break
            seen.add(row["dst_id"])
            cur = self.get(row["dst_id"])
            chain.append(cur)
        return chain

    # ------------------------------------------------------------------
    # preregistration
    # ------------------------------------------------------------------
    def preregister(self, ref: Any, expectation: Expectation) -> int:
        """Seal an expectation. Returns the prereg id to pass to record()."""
        h = self.get(ref)
        sealed_at, seal_hash = expectation.seal()
        cur = self.conn.execute(
            "INSERT INTO preregs (hypothesis_id, expectation_json, sealed_at, seal_hash)"
            " VALUES (?,?,?,?)", (h.id, expectation.to_json(), sealed_at, seal_hash))
        self.conn.commit()
        return int(cur.lastrowid)

    def open_prereg(self, ref: Any) -> Optional[Tuple[int, Expectation]]:
        """The most recent unconsumed sealed expectation for a hypothesis."""
        h = self.get(ref)
        row = self.conn.execute(
            "SELECT * FROM preregs WHERE hypothesis_id=? AND consumed=0"
            " ORDER BY id DESC LIMIT 1", (h.id,)).fetchone()
        if not row:
            return None
        exp = Expectation.from_json(row["expectation_json"])
        if not exp.verify(row["sealed_at"], row["seal_hash"]):
            raise RuntimeError(
                f"prereg {row['id']} failed its seal check -- the database was tampered with")
        return int(row["id"]), exp

    def verify_seals(self) -> List[int]:
        """Ids of any preregistrations whose contents no longer match their hash."""
        bad = []
        for row in self.conn.execute("SELECT * FROM preregs"):
            exp = Expectation.from_json(row["expectation_json"])
            if not exp.verify(row["sealed_at"], row["seal_hash"]):
                bad.append(int(row["id"]))
        return bad

    # ------------------------------------------------------------------
    # experiments
    # ------------------------------------------------------------------
    def record(self, ref: Any, verdict: str, metrics: Dict[str, float],
               config: Optional[Dict[str, Any]] = None,
               regimes: Optional[Dict[str, Dict[str, float]]] = None,
               checks: Optional[List[Dict[str, Any]]] = None,
               prereg_id: Optional[int] = None, data_fingerprint: str = "",
               code_hash: str = "", n_trials: int = 1, notes: str = "",
               update_status: bool = True) -> int:
        """Append a result. Nothing here can ever be edited or removed."""
        h = self.get(ref)
        cur = self.conn.execute(
            "INSERT INTO experiments (hypothesis_id, prereg_id, verdict, config_json,"
            " metrics_json, regime_json, checks_json, data_fingerprint, code_hash,"
            " n_trials, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (h.id, prereg_id, verdict, json.dumps(config or {}),
             json.dumps(metrics), json.dumps(regimes or {}), json.dumps(checks or []),
             data_fingerprint, code_hash, n_trials, notes, _now()))
        if prereg_id is not None:
            self.conn.execute("UPDATE preregs SET consumed=1 WHERE id=?", (prereg_id,))
        if update_status:
            self.conn.execute("UPDATE hypotheses SET status=?, updated_at=? WHERE id=?",
                              (verdict, _now(), h.id))
        self.conn.commit()
        return int(cur.lastrowid)

    def experiments(self, ref: Any = None) -> List[Experiment]:
        if ref is None:
            rows = self.conn.execute("SELECT * FROM experiments ORDER BY id")
        else:
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE hypothesis_id=? ORDER BY id",
                (self.get(ref).id,))
        return [Experiment(
            id=r["id"], hypothesis_id=r["hypothesis_id"], verdict=r["verdict"],
            config=json.loads(r["config_json"]), metrics=json.loads(r["metrics_json"]),
            regimes=json.loads(r["regime_json"]), checks=json.loads(r["checks_json"]),
            data_fingerprint=r["data_fingerprint"], code_hash=r["code_hash"],
            n_trials=r["n_trials"], notes=r["notes"], created_at=r["created_at"],
        ) for r in rows]

    # ------------------------------------------------------------------
    # lessons
    # ------------------------------------------------------------------
    def add_lesson(self, kind: str, text: str, ref: Any = None,
                   experiment_id: Optional[int] = None, regime: str = "",
                   evidence: Optional[Dict[str, Any]] = None) -> None:
        hid = self.get(ref).id if ref is not None else None
        self.conn.execute(
            "INSERT OR IGNORE INTO lessons (hypothesis_id, experiment_id, kind,"
            " regime, text, evidence_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (hid, experiment_id, kind, regime, text,
             json.dumps(evidence or {}), _now()))
        self.conn.commit()

    def lessons(self, ref: Any = None, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        q, args = "SELECT * FROM lessons", []
        where = []
        if ref is not None:
            where.append("hypothesis_id=?")
            args.append(self.get(ref).id)
        if kind:
            where.append("kind=?")
            args.append(kind)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY id DESC"
        return [{"id": r["id"], "hypothesis_id": r["hypothesis_id"],
                 "experiment_id": r["experiment_id"], "kind": r["kind"],
                 "regime": r["regime"], "text": r["text"],
                 "evidence": json.loads(r["evidence_json"]),
                 "created_at": r["created_at"]}
                for r in self.conn.execute(q, args)]

    # ------------------------------------------------------------------
    # the queries that make the graph worth keeping
    # ------------------------------------------------------------------
    def never_tried(self) -> List[Hypothesis]:
        """The frontier. Ideas stated but never run -- what an agent should pick
        up next, and the only list that stops it re-deriving old ground."""
        return [self._hyp(r) for r in self.conn.execute(
            "SELECT h.* FROM hypotheses h LEFT JOIN experiments e"
            " ON e.hypothesis_id = h.id WHERE e.id IS NULL AND h.status != 'retired'"
            " ORDER BY h.id")]

    def refuted_in(self, regime: str) -> List[Tuple[Hypothesis, Dict[str, float]]]:
        """Everything that has been shown to lose money in a given regime --
        including hypotheses that are confirmed overall. This is the list that
        does not exist anywhere else."""
        out = []
        for exp in self.experiments():
            rm = exp.regimes.get(regime)
            if rm and rm.get("sharpe", 0.0) < 0:
                out.append((self.get(exp.hypothesis_id), rm))
        return out

    def confirmed_only_in(self) -> List[Tuple[Hypothesis, List[str]]]:
        """Hypotheses that work, but only somewhere. Kept separate from
        `confirmed` because deploying one of these blind is how you lose money."""
        out = []
        for h in self.all(status=REGIME_CONDITIONAL):
            regimes = []
            for exp in self.experiments(h.id):
                regimes += [lab for lab, rm in exp.regimes.items()
                            if rm.get("sharpe", 0.0) > 0]
            if regimes:
                out.append((h, sorted(set(regimes))))
        return out

    def trial_count(self, tag: Optional[str] = None,
                    family: Optional[str] = None) -> int:
        """How many variants have actually been tested. Feeds the Deflated Sharpe.

        A search that ran 4,000 parameter combinations and reports the best one
        has an n_trials of 4,000, not 1. The graph is the only thing that knows
        this, because it is the only thing that watched every attempt.
        """
        if family:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(e.n_trials),0) AS n FROM experiments e"
                " JOIN hypotheses h ON h.id=e.hypothesis_id"
                " WHERE h.slug LIKE ? OR (','||h.tags||',') LIKE ?",
                (f"{family}%", f"%,{family},%")).fetchone()
        elif tag:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(e.n_trials),0) AS n FROM experiments e"
                " JOIN hypotheses h ON h.id=e.hypothesis_id"
                " WHERE (','||h.tags||',') LIKE ?", (f"%,{tag},%",)).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(n_trials),0) AS n FROM experiments").fetchone()
        return int(row["n"])

    def tested_configs(self) -> Dict[str, List[Dict[str, Any]]]:
        """code_hash -> configs already run, so an agent can skip exact repeats."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        for exp in self.experiments():
            out.setdefault(exp.code_hash, []).append(exp.config)
        return out

    def already_tested(self, code_hash: str, data_fingerprint: str = "") -> bool:
        q = "SELECT 1 FROM experiments WHERE code_hash=?"
        args: List[Any] = [code_hash]
        if data_fingerprint:
            q += " AND data_fingerprint=?"
            args.append(data_fingerprint)
        return self.conn.execute(q, args).fetchone() is not None

    def similar(self, statement: str, k: int = 5,
                exclude: Optional[Iterable[int]] = None
                ) -> List[Tuple[Hypothesis, float]]:
        """Nearest prior hypotheses by TF-IDF cosine over statement + tags.

        No embedding service, no API key, no network. Good enough to catch "you
        already tried this, phrased differently", which is the job.
        """
        docs = self.all()
        if not docs:
            return []
        skip = set(exclude or ())
        corpus = [(h, _tokens(h.statement + " " + " ".join(h.tags))) for h in docs]
        df = Counter()
        for _, toks in corpus:
            df.update(set(toks))
        n = len(corpus)

        def vec(toks: List[str]) -> Dict[str, float]:
            tf = Counter(toks)
            # +1 keeps terms that appear in every document from zeroing out,
            # which matters constantly on a graph with a handful of rows.
            return {t: (c / len(toks)) * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1.0)
                    for t, c in tf.items()}

        qv = vec(_tokens(statement))
        scored = []
        for h, toks in corpus:
            if h.id in skip:
                continue
            dv = vec(toks)
            num = sum(qv.get(t, 0.0) * dv.get(t, 0.0) for t in set(qv) | set(dv))
            den = (math.sqrt(sum(v * v for v in qv.values()))
                   * math.sqrt(sum(v * v for v in dv.values())))
            if den:
                scored.append((h, num / den))
        scored.sort(key=lambda t: -t[1])
        return [(h, s) for h, s in scored[:k] if s > 0]

    def summary(self) -> Dict[str, Any]:
        counts = {s: 0 for s in STATUSES}
        for row in self.conn.execute(
                "SELECT status, COUNT(*) c FROM hypotheses GROUP BY status"):
            counts[row["status"]] = row["c"]
        one = lambda q: int(self.conn.execute(q).fetchone()[0])
        return {
            "path": self.path,
            "hypotheses": sum(counts.values()),
            "by_status": counts,
            "experiments": one("SELECT COUNT(*) FROM experiments"),
            "trials": self.trial_count(),
            "lessons": one("SELECT COUNT(*) FROM lessons"),
            "edges": one("SELECT COUNT(*) FROM edges"),
            "preregs": one("SELECT COUNT(*) FROM preregs"),
            "open_preregs": one("SELECT COUNT(*) FROM preregs WHERE consumed=0"),
            "seal_failures": len(self.verify_seals()),
        }

    def export(self) -> Dict[str, Any]:
        """Whole graph as JSON -- for diffing, sharing or feeding to a model."""
        return {
            "hypotheses": [h.to_dict() for h in self.all()],
            "edges": [dict(r) for r in self.conn.execute("SELECT * FROM edges")],
            "experiments": [vars(e) for e in self.experiments()],
            "lessons": self.lessons(),
            "summary": self.summary(),
        }


_STOP = {"the", "a", "an", "is", "in", "on", "of", "and", "or", "to", "for",
         "with", "that", "this", "it", "as", "at", "by", "be", "are", "will",
         "when", "than", "then", "from", "into", "over", "under", "does"}


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOP and len(t) > 1]
