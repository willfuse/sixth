"""The live boundary: a paper broker and a risk gate.

The article states the rule this module exists to enforce:

  "The kill switch lives in code, never in the system prompt. A risk limit
   written into a prompt is a suggestion, and a sufficiently motivated reasoning
   chain will argue its way past it."

So `RiskGate` is not advice to a model. It is a function that returns a smaller
number, or zero, and there is no argument an agent can make to it. Every order
in this package passes through it, and the gate has no text interface at all.

Scope: this ships a paper broker only. It does not route real orders and holds no
venue credentials. Connecting a live broker is deliberately left to you -- see
`BrokerAdapter` and read the whole class before you implement it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class KillSwitchTripped(RuntimeError):
    """Raised when a hard limit is breached. Trading stops; it does not degrade."""


@dataclass
class RiskLimits:
    """Hard bounds. Every one of these is checked in code on every order."""

    max_position: float = 1.0          # absolute target weight per symbol
    max_gross_exposure: float = 1.0    # sum of |weights| across the book
    max_daily_loss: float = 0.02       # fraction of equity, then stop for the day
    max_drawdown: float = 0.15         # fraction from peak, then stop entirely
    max_orders_per_day: int = 50
    max_order_notional: float = 0.25   # fraction of equity in one order
    allow_shorts: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(vars(self))


@dataclass
class Order:
    symbol: str
    target_weight: float
    reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Fill:
    symbol: str
    weight_before: float
    weight_after: float
    price: float
    cost: float
    at: str


class RiskGate:
    """Clamps or refuses orders. Stateful, deterministic, unpersuadable.

    `check` returns the weight actually permitted. It never raises for an ordinary
    clamp -- it just returns less. It raises only when a kill-switch condition is
    hit, because at that point the correct behaviour is to stop, not to trade
    smaller.
    """

    def __init__(self, limits: Optional[RiskLimits] = None, equity: float = 1.0):
        self.limits = limits or RiskLimits()
        self.start_equity = equity
        self.equity = equity
        self.peak_equity = equity
        self.day_start_equity = equity
        self.day: Optional[str] = None
        self.orders_today = 0
        self.halted = False
        self.halt_reason = ""
        self.journal: List[Dict[str, Any]] = []

    # -- state ---------------------------------------------------------
    def mark(self, equity: float, day: Optional[str] = None) -> None:
        """Update equity, roll the day, and trip the switch if a bound is broken."""
        if day is not None and day != self.day:
            self.day = day
            self.day_start_equity = equity
            self.orders_today = 0
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity else 0.0
        if dd > self.limits.max_drawdown:
            self._halt(f"max drawdown breached: {dd:.2%} > {self.limits.max_drawdown:.2%}")

        day_loss = ((self.day_start_equity - equity) / self.day_start_equity
                    if self.day_start_equity else 0.0)
        if day_loss > self.limits.max_daily_loss:
            self._halt(f"daily loss breached: {day_loss:.2%} > {self.limits.max_daily_loss:.2%}")

    def _halt(self, reason: str) -> None:
        if not self.halted:
            self.halted = True
            self.halt_reason = reason
            self._log("HALT", reason=reason)

    def _log(self, event: str, **kw: Any) -> None:
        self.journal.append({"at": datetime.now(timezone.utc).isoformat(),
                             "event": event, "equity": self.equity, **kw})

    # -- the gate ------------------------------------------------------
    def check(self, order: Order, current_weight: float = 0.0,
              book: Optional[Dict[str, float]] = None) -> float:
        """Return the permitted target weight for this order.

        Raises KillSwitchTripped when trading must stop entirely.
        """
        if self.halted:
            raise KillSwitchTripped(self.halt_reason)

        if self.orders_today >= self.limits.max_orders_per_day:
            self._halt(f"order count limit reached: {self.limits.max_orders_per_day}/day")
            raise KillSwitchTripped(self.halt_reason)

        w = float(order.target_weight)

        if not self.limits.allow_shorts and w < 0:
            self._log("CLAMP", symbol=order.symbol, was=w, now=0.0, rule="shorts_disabled")
            w = 0.0

        capped = max(-self.limits.max_position, min(self.limits.max_position, w))
        if capped != w:
            self._log("CLAMP", symbol=order.symbol, was=w, now=capped, rule="max_position")
            w = capped

        if abs(w - current_weight) > self.limits.max_order_notional:
            step = self.limits.max_order_notional * (1 if w > current_weight else -1)
            self._log("CLAMP", symbol=order.symbol, was=w, now=current_weight + step,
                      rule="max_order_notional")
            w = current_weight + step

        if book:
            others = sum(abs(v) for k, v in book.items() if k != order.symbol)
            room = self.limits.max_gross_exposure - others
            if abs(w) > room:
                sign = 1 if w >= 0 else -1
                self._log("CLAMP", symbol=order.symbol, was=w, now=sign * max(room, 0.0),
                          rule="max_gross_exposure")
                w = sign * max(room, 0.0)

        self.orders_today += 1
        return w

    def report(self) -> Dict[str, Any]:
        return {
            "halted": self.halted, "halt_reason": self.halt_reason,
            "equity": self.equity, "peak_equity": self.peak_equity,
            "return_since_start": self.equity / self.start_equity - 1.0,
            "orders_today": self.orders_today,
            "limits": self.limits.to_dict(),
            "journal_entries": len(self.journal),
        }

    def write_journal(self, path: str) -> None:
        """Every decision with the reason that produced it -- the input the
        post-mortem reads."""
        with open(path, "w") as fh:
            for row in self.journal:
                fh.write(json.dumps(row) + "\n")


class BrokerAdapter:
    """Interface for a venue. Implement this yourself if you go live.

    `sixth` ships no live implementation and stores no credentials, on purpose.
    Two things to keep true in anything you write against this:

      1. Route every order through `RiskGate.check` and trade its return value,
         never the model's requested weight.
      2. Reconcile against the broker's own position report each cycle. Local
         state drifting from venue state is the failure that turns a bad day into
         an unbounded one.
    """

    def positions(self) -> Dict[str, float]:
        raise NotImplementedError

    def equity(self) -> float:
        raise NotImplementedError

    def submit(self, order: Order, permitted_weight: float) -> Fill:
        raise NotImplementedError


class PaperBroker(BrokerAdapter):
    """Simulated fills against a supplied price. No network, no venue, no money."""

    def __init__(self, equity: float = 100_000.0, cost_bps: float = 3.0):
        self._equity = equity
        self._positions: Dict[str, float] = {}
        self.cost_bps = cost_bps
        self.fills: List[Fill] = []

    def positions(self) -> Dict[str, float]:
        return dict(self._positions)

    def equity(self) -> float:
        return self._equity

    def submit(self, order: Order, permitted_weight: float,
               price: float = 100.0) -> Fill:
        before = self._positions.get(order.symbol, 0.0)
        traded = abs(permitted_weight - before)
        cost = traded * self.cost_bps / 10_000.0 * self._equity
        self._equity -= cost
        self._positions[order.symbol] = permitted_weight
        fill = Fill(order.symbol, before, permitted_weight, price, cost,
                    datetime.now(timezone.utc).isoformat())
        self.fills.append(fill)
        return fill

    def apply_return(self, symbol: str, ret: float) -> None:
        """Mark the book forward by one bar's return on the held weight."""
        self._equity *= (1.0 + self._positions.get(symbol, 0.0) * ret)
