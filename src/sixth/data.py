"""Bar data: a CSV loader and a deterministic synthetic generator.

The synthetic generator exists so the whole loop runs offline with no vendor
account, and so the test suite has a market whose regimes are known ground truth.
Point `load_csv` at real bars when you have them.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence


@dataclass
class Bars:
    """OHLCV series for one symbol. Index-aligned lists, oldest first."""

    symbol: str
    dates: List[str]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.close)

    def __post_init__(self) -> None:
        n = len(self.close)
        for name in ("dates", "open", "high", "low"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has length {len(getattr(self, name))}, close has {n}")
        if not self.volume:
            self.volume = [0.0] * n

    def returns(self) -> List[float]:
        """Simple close-to-close returns; first element is 0.0."""
        out = [0.0]
        for i in range(1, len(self.close)):
            prev = self.close[i - 1]
            out.append((self.close[i] - prev) / prev if prev else 0.0)
        return out

    def slice(self, start: int, end: Optional[int] = None) -> "Bars":
        end = len(self) if end is None else end
        return Bars(
            symbol=self.symbol,
            dates=self.dates[start:end], open=self.open[start:end],
            high=self.high[start:end], low=self.low[start:end],
            close=self.close[start:end], volume=self.volume[start:end],
        )

    def fingerprint(self) -> str:
        """Content hash. Stored with every experiment so a result can never be
        silently compared against a different dataset."""
        h = hashlib.sha256()
        h.update(self.symbol.encode())
        for d, c in zip(self.dates, self.close):
            h.update(f"{d}:{c:.10g}|".encode())
        return h.hexdigest()[:16]


def load_csv(path: str, symbol: Optional[str] = None,
             date_col: str = "date", close_col: str = "close") -> Bars:
    """Load bars from CSV. Missing OHLC columns fall back to close."""
    dates, o, h, lo, c, v = [], [], [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            keys = {k.lower().strip(): k for k in row}

            def get(name: str, default: Optional[float] = None) -> Optional[float]:
                k = keys.get(name)
                if k is None or row[k] in ("", None):
                    return default
                return float(row[k])

            close = get(close_col)
            if close is None:
                raise ValueError(f"{path}: no '{close_col}' column")
            dates.append(row[keys[date_col]])
            c.append(close)
            o.append(get("open", close))
            h.append(get("high", close))
            lo.append(get("low", close))
            v.append(get("volume", 0.0))
    return Bars(symbol or path.rsplit("/", 1)[-1].split(".")[0], dates, o, h, lo, c, v)


# --------------------------------------------------------------------------
# synthetic market with known regimes
# --------------------------------------------------------------------------
REGIME_SPECS: Dict[str, Dict[str, float]] = {
    #                  drift/day   vol/day   mean-reversion pull
    "bull_quiet":     {"mu": 0.0006, "sigma": 0.008, "phi": 0.02},
    "bull_volatile":  {"mu": 0.0008, "sigma": 0.020, "phi": 0.00},
    "chop":           {"mu": 0.0000, "sigma": 0.010, "phi": 0.25},
    "bear_grind":     {"mu": -0.0004, "sigma": 0.013, "phi": 0.05},
    "crash":          {"mu": -0.0035, "sigma": 0.035, "phi": -0.10},
}


def synthetic_bars(symbol: str = "SYNTH", n: int = 2520, seed: int = 7,
                   start: str = "2016-01-04",
                   regimes: Optional[Sequence[str]] = None) -> Bars:
    """A regime-switching price series. Deterministic given `seed`.

    Not a claim about real markets -- a fixture with trends, chop and crashes in
    known places, so a backtester can be checked against ground truth.
    """
    rng = random.Random(seed)
    names = list(regimes) if regimes else list(REGIME_SPECS)
    labels = _regime_path(n, names, rng)

    price = 100.0
    prev_ret = 0.0
    d = date.fromisoformat(start)
    dates, o, h, lo, c, v = [], [], [], [], [], []
    for i in range(n):
        spec = REGIME_SPECS[labels[i]]
        shock = rng.gauss(0.0, spec["sigma"])
        ret = spec["mu"] + shock - spec["phi"] * prev_ret
        prev_ret = ret
        open_p = price
        price = max(price * (1.0 + ret), 0.01)
        wick = abs(rng.gauss(0.0, spec["sigma"] * 0.5)) * price
        dates.append(d.isoformat())
        o.append(round(open_p, 4))
        c.append(round(price, 4))
        h.append(round(max(open_p, price) + wick, 4))
        lo.append(round(max(min(open_p, price) - wick, 0.01), 4))
        v.append(round(1e6 * math.exp(rng.gauss(0, 0.3)), 0))
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
    return Bars(symbol, dates, o, h, lo, c, v)


def _regime_path(n: int, names: Sequence[str], rng: random.Random) -> List[str]:
    """Persistent regimes: pick one, hold it for a while, then switch."""
    labels: List[str] = []
    while len(labels) < n:
        name = rng.choice(names)
        hold = int(rng.uniform(60, 400)) if name != "crash" else int(rng.uniform(10, 45))
        labels.extend([name] * hold)
    return labels[:n]


def synthetic_true_regimes(n: int = 2520, seed: int = 7,
                           regimes: Optional[Sequence[str]] = None) -> List[str]:
    """The generator's own regime labels -- ground truth for testing the
    detector in regimes.py."""
    rng = random.Random(seed)
    names = list(regimes) if regimes else list(REGIME_SPECS)
    return _regime_path(n, names, rng)
