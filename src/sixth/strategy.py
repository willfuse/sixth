"""Strategy interface.

A strategy maps bars -> target weight per bar, in [-1, 1]. That is the whole
contract. It is deliberately tiny because, as the article puts it, "strategy code
is small, well specified and easy to test, which makes it close to the perfect
task for a coding agent" -- and a narrow contract is what makes generated code
safe to run unattended.

Rule enforced here: `weights()` returns the target for bar i using data up to and
including bar i, and the engine trades it at bar i+1's open. Lookahead is
structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from .data import Bars

REGISTRY: Dict[str, "Strategy"] = {}


@dataclass
class Strategy:
    name: str
    fn: Callable[..., Sequence[float]]
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def weights(self, bars: Bars) -> List[float]:
        w = list(self.fn(bars, **self.params))
        if len(w) != len(bars):
            raise ValueError(
                f"{self.name}: returned {len(w)} weights for {len(bars)} bars")
        return [max(-1.0, min(1.0, float(x))) for x in w]

    def code_hash(self) -> str:
        """Hash of source + params. Two experiments with the same hash ran the
        same thing; a changed hash means the graph is looking at a new variant."""
        try:
            src = inspect.getsource(self.fn)
        except (OSError, TypeError):
            src = self.fn.__name__
        payload = src + repr(sorted(self.params.items()))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "params": dict(self.params),
                "code_hash": self.code_hash(), "description": self.description}

    def with_params(self, **overrides: Any) -> "Strategy":
        p = dict(self.params)
        p.update(overrides)
        return Strategy(self.name, self.fn, p, self.description)


def register(name: str, description: str = "", **default_params: Any):
    """Decorator: make a weight function available to the CLI by name."""
    def deco(fn: Callable[..., Sequence[float]]) -> Callable[..., Sequence[float]]:
        REGISTRY[name] = Strategy(name, fn, dict(default_params), description)
        return fn
    return deco


def get(name: str, **params: Any) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; have: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name].with_params(**params) if params else REGISTRY[name]


# --------------------------------------------------------------------------
# built-ins -- reference implementations, and the null models the breaker needs
# --------------------------------------------------------------------------
def _sma(xs: Sequence[float], window: int) -> List[float]:
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= window:
            acc -= xs[i - window]
        out.append(acc / min(i + 1, window))
    return out


@register("sma_cross", "Long when fast SMA is above slow SMA.", fast=20, slow=100)
def sma_cross(bars: Bars, fast: int = 20, slow: int = 100) -> List[float]:
    f, s = _sma(bars.close, fast), _sma(bars.close, slow)
    return [0.0 if i < slow else (1.0 if f[i] > s[i] else 0.0) for i in range(len(bars))]


@register("sma_cross_ls", "Long/short SMA cross.", fast=20, slow=100)
def sma_cross_ls(bars: Bars, fast: int = 20, slow: int = 100) -> List[float]:
    f, s = _sma(bars.close, fast), _sma(bars.close, slow)
    return [0.0 if i < slow else (1.0 if f[i] > s[i] else -1.0) for i in range(len(bars))]


@register("momentum", "Long if trailing return over `lookback` is positive.",
          lookback=60)
def momentum(bars: Bars, lookback: int = 60) -> List[float]:
    c = bars.close
    out = []
    for i in range(len(c)):
        if i < lookback:
            out.append(0.0)
        else:
            out.append(1.0 if c[i] > c[i - lookback] else 0.0)
    return out


@register("mean_reversion", "Fade moves beyond `z` trailing standard deviations.",
          window=20, z=1.0)
def mean_reversion(bars: Bars, window: int = 20, z: float = 1.0) -> List[float]:
    from .regimes import rolling_std
    c = bars.close
    ma = _sma(c, window)
    sd = rolling_std(c, window)
    out = []
    for i in range(len(c)):
        if i < window or sd[i] == 0:
            out.append(0.0)
            continue
        dev = (c[i] - ma[i]) / sd[i]
        out.append(-1.0 if dev > z else (1.0 if dev < -z else 0.0))
    return out


@register("buy_hold", "Always long. The benchmark every strategy must beat.")
def buy_hold(bars: Bars) -> List[float]:
    return [1.0] * len(bars)


@register("flat", "Never in the market. Null model.")
def flat(bars: Bars) -> List[float]:
    return [0.0] * len(bars)


@register("random_signal", "Coin-flip positions. The null the breaker compares to.",
          seed=0, p_long=0.5)
def random_signal(bars: Bars, seed: int = 0, p_long: float = 0.5) -> List[float]:
    import random as _r
    rng = _r.Random(seed)
    return [1.0 if rng.random() < p_long else 0.0 for _ in range(len(bars))]


#: Public alias. Inside the package `get` is unambiguous; from outside,
#: `sixth.get_strategy` avoids colliding with the `sixth.strategy` module.
get_strategy = get
