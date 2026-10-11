#!/usr/bin/env python3
"""Offline, resumable 510300 daily replay using the unmodified Stonkfly brain.

The fly sees closes through day T, then its decision is executed at day T+1 open.
This is a paper experiment, not an order-routing program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

from PIL import Image, ImageDraw

from stonkfly.config import Settings
from stonkfly.display import market_frame
from stonkfly.neural.controller import FlyController


D = Decimal
LOT = 100
SYMBOL = "510300.SH"
POLICY_VERSION = "a-share-action-credit-v3"
REVERSAL_COOLDOWN_DAYS = 3


def _side_code(side: str | None) -> float:
    return {"SELL": -1.0, "HOLD": 0.0, "NONE": 0.0, "BUY": 1.0}.get(
        str(side or "NONE").upper(), 0.0
    )


def reversal_cooldown(last_fill_side: str | None, last_fill_index: int | None, side: str, local_index: int) -> bool:
    """Return whether an opposite BUY/SELL request is inside the inclusive cooldown."""
    if last_fill_side not in {"BUY", "SELL"} or side not in {"BUY", "SELL"}:
        return False
    if side == last_fill_side or last_fill_index is None:
        return False
    elapsed = local_index - last_fill_index
    return 0 <= elapsed <= REVERSAL_COOLDOWN_DAYS


def action_advantage(side: str, quantity: int, close: Decimal, fill: Decimal, fee: Decimal) -> Decimal:
    """Mark a filled action against the close using Decimal arithmetic only."""
    if side == "BUY":
        return D(quantity) * (close - fill) - fee
    if side == "SELL":
        return D(quantity) * (fill - close) - fee
    return D("0")


def action_advantage_sign(value: Decimal) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def days_since_fill(last_fill_index: int | None, local_index: int) -> float:
    """Encode no fill as 10 and cap the elapsed trading days at 10."""
    if last_fill_index is None:
        return 10.0
    return float(max(0, min(10, local_index - last_fill_index)))


def record_veto(state: dict[str, object], reason: str) -> None:
    state["veto_count"] = int(state.get("veto_count", 0)) + 1
    reasons = state.setdefault("veto_reasons", {})
    assert isinstance(reasons, dict)
    reasons[reason] = int(reasons.get(reason, 0)) + 1
    if reason == "reversal_cooldown":
        state["reversal_count"] = int(state.get("reversal_count", 0)) + 1
        # Keep the explicit old name as a compatibility alias in checkpoints.
        state["reversal_veto_count"] = int(state.get("reversal_veto_count", 0)) + 1


def default_state(signature: str, capital: Decimal) -> dict[str, object]:
    return {
        "signature": signature,
        "next_index": 0,
        "cash": str(capital),
        "shares": 0,
        "sellable_shares": 0,
        "unsettled": [],
        "anchor": str(capital),
        "peak_equity": str(capital),
        "drawdown": 0.0,
        "position_cost": "0",
        "pending_signal": "HOLD",
        "last_fill_side": None,
        "last_fill_index": None,
        "action_advantage_cny": "0",
        "action_advantage_ratio": 0.0,
        "action_advantage_sign": 0.0,
        "last_portfolio_observation": {},
        "fees": "0",
        "buy_count": 0,
        "sell_count": 0,
        "veto_count": 0,
        "veto_reasons": {},
        "reversal_count": 0,
        "reversal_veto_count": 0,
        "action_counts": {},
        "equity_curve": [],
        "checkpoint": None,
    }


def restore_state(state: dict[str, object], signature: str, capital: Decimal) -> dict[str, object]:
    """Fill fields introduced by v3 before the first resumed step."""
    defaults = default_state(signature, capital)
    for key, value in defaults.items():
        state.setdefault(key, value.copy() if isinstance(value, (dict, list)) else value)
    state["signature"] = signature
    if not state.get("reversal_count") and state.get("reversal_veto_count"):
        state["reversal_count"] = state["reversal_veto_count"]
    if not state.get("reversal_veto_count") and state.get("reversal_count"):
        state["reversal_veto_count"] = state["reversal_count"]
    return state

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def load_prices(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item: dict[str, object] = {"date": row["date"]}
            for field in ("open", "high", "low", "close"):
                value = D(row[field])
                if not value.is_finite() or value <= 0:
                    raise ValueError(f"invalid {field} on {row['date']}")
                item[field] = value
            item["volume"] = int(float(row["volume"]))
            rows.append(item)
    if not rows or [r["date"] for r in rows] != sorted(r["date"] for r in rows):
        raise ValueError("price rows must be nonempty and sorted")
    return rows


def load_dividends(path: Path) -> dict[str, Decimal]:
    cumulative = D("0")
    result: dict[str, Decimal] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            current = D(row["累计分红"])
            amount = current - cumulative
            if amount < 0:
                raise ValueError("cumulative dividends went backwards")
            if amount:
                result[row["日期"]] = amount
            cumulative = current
    return result


def commission(notional: Decimal, rate: Decimal, minimum: Decimal) -> Decimal:
    return max(minimum, notional * rate).quantize(D("0.01"))


def max_drawdown(values: list[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def buy_capacity(
    cash: Decimal,
    shares: int,
    price: Decimal,
    order_notional: Decimal,
    max_exposure: Decimal,
) -> Decimal:
    """Notional budget that cannot push the position above the exposure guard."""
    equity = cash + price * shares
    room = max(D("0"), equity * max_exposure - price * shares)
    return min(order_notional, cash, room)


def shaped_feedback(
    *,
    equity: Decimal,
    previous_equity: Decimal,
    benchmark_equity: Decimal,
    previous_benchmark_equity: Decimal,
    drawdown: float,
    previous_drawdown: float,
    turnover_ratio: float,
    vetoed: bool,
    deadband_bps: float,
    drawdown_weight: float,
    turnover_weight: float,
    veto_penalty_bps: float,
    action_advantage_ratio: float = 0.0,
    action_advantage_weight: float = 1.0,
) -> dict[str, object]:
    """Turn relative return and risk changes into an engineered DAN stimulus."""
    portfolio_return = float(equity / previous_equity - D("1"))
    benchmark_return = float(
        benchmark_equity / previous_benchmark_equity - D("1")
    )
    excess_return = portfolio_return - benchmark_return
    drawdown_change = drawdown - previous_drawdown
    score = excess_return + drawdown_weight * drawdown_change + action_advantage_weight * action_advantage_ratio
    score -= turnover_weight * turnover_ratio
    if vetoed:
        score -= veto_penalty_bps / 10_000
    threshold = deadband_bps / 10_000
    stimulus = (
        "reward"
        if score >= threshold
        else "aversive"
        if score <= -threshold
        else "none"
    )
    return {
        "stimulus": stimulus,
        "score": score,
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "excess_return": excess_return,
        "drawdown_change": drawdown_change,
        "turnover_ratio": turnover_ratio,
        "veto_penalty_applied": vetoed,
        "action_advantage_ratio": action_advantage_ratio,
    }


def portfolio_frame(
    frame,
    *,
    exposure: float,
    cash_ratio: float,
    unrealized_return: float,
    drawdown: float,
    relative_return: float,
    buy_capacity_ratio: float = 0.0,
    sellable_ratio: float = 0.0,
    last_fill_side: str = "NONE",
    days_since_fill: float = 10.0,
    action_advantage_sign: float = 0.0,
    action_advantage_ratio: float = 0.0,
    vetoed: bool = False,
):
    """Add a fixed RGB status strip; the neural graph still receives RGB only."""
    import numpy as np

    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 28, 319, 53), fill=(12, 22, 43))
    # Eleven compact columns keep every v3 status value inside the 320px frame.
    metrics = [
        ("E", exposure, 0.0, 1.0),
        ("C", cash_ratio, 0.0, 1.0),
        ("P", unrealized_return, -0.25, 0.25),
        ("D", drawdown, -0.25, 0.0),
        ("A", relative_return, -0.15, 0.15),
        ("B", buy_capacity_ratio, 0.0, 1.0),
        ("S", sellable_ratio, 0.0, 1.0),
        ("L", days_since_fill, 0.0, 10.0),
        ("F", _side_code(last_fill_side), -1.0, 1.0),
        ("X", action_advantage_sign, -1.0, 1.0),
        ("R", action_advantage_ratio, -0.25, 0.25),
    ]
    for index, (label, value, low, high) in enumerate(metrics):
        left = index * 29
        right = left + 27
        normalized = max(0.0, min(1.0, (value - low) / (high - low)))
        fill = (44, 183, 116) if value >= 0 else (224, 73, 93)
        draw.text((left + 1, 30), label, fill=(220, 230, 245))
        draw.rectangle((left + 8, 33, right, 46), outline=(94, 111, 143))
        width = round((right - left - 9) * normalized)
        if width:
            draw.rectangle((left + 9, 34, left + 8 + width, 45), fill=fill)
        draw.text((left + 8, 46), f"{value:+.1%}"[:5], fill=(180, 194, 220))
    if vetoed:
        draw.rectangle((0, 28, 319, 53), outline=(255, 55, 55), width=2)
    return np.asarray(image, dtype=np.uint8)

def buy_and_hold(
    rows: list[dict[str, object]],
    dividends: dict[str, Decimal],
    capital: Decimal,
    rate: Decimal,
    minimum: Decimal,
    slippage: Decimal,
) -> tuple[dict[str, object], list[Decimal]]:
    price = rows[0]["open"] * (D("1") + slippage)
    quantity = int((capital / price / LOT).to_integral_value(rounding=ROUND_DOWN)) * LOT
    while quantity:
        notional = price * quantity
        fee = commission(notional, rate, minimum)
        if notional + fee <= capital:
            break
        quantity -= LOT
    fee = commission(price * quantity, rate, minimum) if quantity else D("0")
    cash = capital - price * quantity - fee
    curve: list[Decimal] = []
    for index, row in enumerate(rows):
        if index:
            cash += dividends.get(str(row["date"]), D("0")) * quantity
        curve.append(cash + row["close"] * quantity)
    final_equity = curve[-1]
    return {
        "quantity": quantity,
        "fee_cny": str(fee),
        "final_equity_cny": str(final_equity),
        "return": float(final_equity / capital - D("1")),
        "max_drawdown": max_drawdown([float(value) for value in curve]),
    }, curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--dividends", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2026-01-16")
    parser.add_argument("--end", default="2026-07-16")
    parser.add_argument("--capital", default="100000")
    parser.add_argument("--order-notional", default="10000")
    parser.add_argument("--commission-rate", default="0.0003")
    parser.add_argument("--minimum-commission", default="5")
    parser.add_argument("--slippage", default="0.0005")
    parser.add_argument("--reward-deadband-bps", type=float, default=1.0)
    parser.add_argument("--drawdown-weight", type=float, default=0.25)
    parser.add_argument("--turnover-weight", type=float, default=0.001)
    parser.add_argument("--veto-penalty-bps", type=float, default=2.0)
    parser.add_argument("--action-advantage-weight", type=float, default=1.0)
    parser.add_argument("--max-exposure", default="0.85")
    parser.add_argument("--initial-brain", type=Path)
    parser.add_argument("--freeze-learning", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    prices_hash = sha256(args.prices)
    rows_all = load_prices(args.prices)
    dividends = load_dividends(args.dividends)
    eval_indices = [
        i
        for i, row in enumerate(rows_all)
        if args.start <= str(row["date"]) <= args.end
    ]
    if not eval_indices:
        raise SystemExit("no rows in requested interval")
    first, last = eval_indices[0], eval_indices[-1]
    rows = rows_all[first : last + 1]
    history_prefix = [float(r["close"]) for r in rows_all[max(0, first - 100) : first]]

    capital = D(args.capital)
    order_notional = D(args.order_notional)
    rate = D(args.commission_rate)
    minimum = D(args.minimum_commission)
    slippage = D(args.slippage)
    max_exposure = D(args.max_exposure)
    if not D("0") < max_exposure < D("1"):
        raise SystemExit("--max-exposure must be between zero and one")
    for name in (
        "reward_deadband_bps",
        "drawdown_weight",
        "turnover_weight",
        "veto_penalty_bps",
        "action_advantage_weight",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and nonnegative")
    initial_brain_hash = sha256(args.initial_brain) if args.initial_brain else None
    financial_config = {
        "capital_cny": str(capital),
        "order_notional_cny": str(order_notional),
        "commission_rate": str(rate),
        "minimum_commission_cny": str(minimum),
        "slippage_each_side": str(slippage),
        "stamp_duty": "0 (domestic equity ETF assumption)",
        "buy_lot": LOT,
        "settlement": "T+1 sell availability",
        "signal_timing": "close T observation; open T+1 execution",
        "max_exposure": str(max_exposure),
    }
    feedback_config = {
        "policy_version": POLICY_VERSION,
        "reward_deadband_bps": args.reward_deadband_bps,
        "drawdown_weight": args.drawdown_weight,
        "turnover_weight": args.turnover_weight,
        "veto_penalty_bps": args.veto_penalty_bps,
        "action_advantage_weight": args.action_advantage_weight,
        "portfolio_visual_feedback": [
            "exposure",
            "cash_ratio",
            "unrealized_return",
            "drawdown",
            "relative_return",
            "buy_capacity_ratio",
            "sellable_ratio",
            "last_fill_side",
            "days_since_fill",
            "action_advantage_sign",
            "action_advantage_ratio",
            "veto",
        ],
        "reversal_cooldown_days": REVERSAL_COOLDOWN_DAYS,
        "reversal_cooldown_boundary": "inclusive local_index - last_fill_index <= 3",
        "action_advantage_formula": {
            "BUY": "qty * (close - fill) - fee",
            "SELL": "qty * (fill - close) - fee",
            "applies_when": "execution.status == FILLED",
            "ratio_denominator": "previous_equity",
        },
    }
    signature_payload = {
        "symbol": SYMBOL,
        "prices_sha256": prices_hash,
        "start": args.start,
        "end": args.end,
        "financial": financial_config,
        "feedback": feedback_config,
        "neural": {"neural_ms": 500, "learning": not args.freeze_learning},
        "initial_brain_sha256": initial_brain_hash,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()

    state_path = args.out / "state.json"
    events_path = args.out / "events.jsonl"
    settings = Settings(learning=not args.freeze_learning)
    controller = FlyController(settings)
    baseline, baseline_curve = buy_and_hold(
        rows, dividends, capital, rate, minimum, slippage
    )
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["signature"] != signature:
            raise SystemExit("run configuration or source data changed; use a new --out")
        state = restore_state(state, signature, capital)
        checkpoint = args.out / state["checkpoint"]["file"]
        if sha256(checkpoint) != state["checkpoint"]["sha256"]:
            raise SystemExit("checkpoint hash mismatch")
        controller.restore(checkpoint)
        controller.brain.weights_frozen = not settings.learning
    else:
        if args.initial_brain:
            controller.restore(args.initial_brain)
            controller.brain.weights_frozen = not settings.learning
        state = default_state(signature, capital)
        atomic_json(
            args.out / "provenance.json",
            {**signature_payload, "signature": signature},
        )

    completed_now = 0
    for local_index in range(int(state["next_index"]), len(rows)):
        if args.max_steps and completed_now >= args.max_steps:
            break
        started = time.monotonic()
        row = rows[local_index]
        date = str(row["date"])
        cash = D(state["cash"])
        shares = int(state["shares"])
        position_cost = D(state["position_cost"])
        sellable = int(state["sellable_shares"])
        unsettled = list(state["unsettled"])

        matured = [lot for lot in unsettled if int(lot["release_index"]) <= local_index]
        sellable += sum(int(lot["quantity"]) for lot in matured)
        unsettled = [lot for lot in unsettled if int(lot["release_index"]) > local_index]

        dividend = dividends.get(date, D("0"))
        dividend_cash = dividend * shares
        cash += dividend_cash
        execution: dict[str, object] = {
            "requested": state["pending_signal"],
            "status": "HOLD",
        }
        turnover_notional = D("0")
        action_adv = D("0")
        pending = str(state["pending_signal"])
        reversal_veto = reversal_cooldown(
            state.get("last_fill_side"),
            state.get("last_fill_index"),
            pending,
            local_index,
        )
        if reversal_veto:
            # The guard runs before any notional/fee/share mutation and rejects
            # the requested reversal without selecting a replacement action.
            record_veto(state, "reversal_cooldown")
            execution = {
                "requested": pending,
                "status": "VETO",
                "reason": "reversal_cooldown",
            }
        if pending == "BUY" and not reversal_veto:
            price = row["open"] * (D("1") + slippage)
            budget = buy_capacity(
                cash, shares, row["open"], order_notional, max_exposure
            )
            quantity = int(
                (budget / price / LOT).to_integral_value(
                    rounding=ROUND_DOWN
                )
            ) * LOT
            while quantity:
                notional = price * quantity
                fee = commission(notional, rate, minimum)
                if notional + fee <= cash:
                    break
                quantity -= LOT
            if quantity:
                notional = price * quantity
                fee = commission(notional, rate, minimum)
                cash -= notional + fee
                shares += quantity
                position_cost += notional + fee
                turnover_notional = notional
                unsettled.append(
                    {"quantity": quantity, "release_index": local_index + 1}
                )
                state["fees"] = str(D(state["fees"]) + fee)
                state["buy_count"] = int(state["buy_count"]) + 1
                execution = {
                    "requested": pending,
                    "status": "FILLED",
                    "quantity": quantity,
                    "price": str(price),
                    "fee": str(fee),
                }
            else:
                record_veto(state, "cash_lot_or_exposure")
                execution = {
                    "requested": pending,
                    "status": "VETO",
                    "reason": "cash_lot_or_exposure",
                }
        elif pending == "SELL" and not reversal_veto:
            price = row["open"] * (D("1") - slippage)
            quantity = min(
                sellable,
                int((order_notional / price / LOT).to_integral_value(rounding=ROUND_DOWN))
                * LOT,
            )
            if quantity:
                shares_before = shares
                notional = price * quantity
                fee = commission(notional, rate, minimum)
                cash += notional - fee
                shares -= quantity
                sellable -= quantity
                position_cost -= position_cost * D(quantity) / D(shares_before)
                turnover_notional = notional
                state["fees"] = str(D(state["fees"]) + fee)
                state["sell_count"] = int(state["sell_count"]) + 1
                execution = {
                    "requested": pending,
                    "status": "FILLED",
                    "quantity": quantity,
                    "price": str(price),
                    "fee": str(fee),
                }
            else:
                record_veto(state, "t_plus_1_or_inventory")
                execution = {
                    "requested": pending,
                    "status": "VETO",
                    "reason": "t_plus_1_or_inventory",
                }

        close = row["close"]
        if execution.get("status") == "FILLED":
            action_adv = action_advantage(
                str(execution["requested"]),
                int(execution["quantity"]),
                close,
                D(execution["price"]),
                D(execution["fee"]),
            )
            state["last_fill_side"] = execution["requested"]
            state["last_fill_index"] = local_index
            state["action_advantage_cny"] = str(
                D(state.get("action_advantage_cny", "0")) + action_adv
            )
        action_adv_ratio = float(action_adv / D(state["anchor"]))
        action_adv_sign = action_advantage_sign(action_adv)
        equity = cash + close * shares
        previous_equity = D(state["anchor"])
        delta = equity - previous_equity
        peak_equity = max(D(state["peak_equity"]), equity)
        drawdown = float(equity / peak_equity - D("1"))
        benchmark_equity = baseline_curve[local_index]
        previous_benchmark = capital if not local_index else baseline_curve[local_index - 1]
        turnover_ratio = float(turnover_notional / previous_equity)
        feedback = shaped_feedback(
            equity=equity,
            previous_equity=previous_equity,
            benchmark_equity=benchmark_equity,
            previous_benchmark_equity=previous_benchmark,
            drawdown=drawdown,
            previous_drawdown=float(state["drawdown"]),
            turnover_ratio=turnover_ratio,
            vetoed=execution["status"] == "VETO",
            deadband_bps=args.reward_deadband_bps,
            drawdown_weight=args.drawdown_weight,
            turnover_weight=args.turnover_weight,
            veto_penalty_bps=args.veto_penalty_bps,
            action_advantage_ratio=float(action_adv / previous_equity),
            action_advantage_weight=args.action_advantage_weight,
        )
        stimulus = str(feedback["stimulus"])
        history = history_prefix + [float(r["close"]) for r in rows[: local_index + 1]]
        frame = market_frame(
            SYMBOL,
            history,
            close * (D("1") - slippage),
            close * (D("1") + slippage),
        )
        exposure = float(close * shares / equity)
        cash_ratio = float(cash / equity)
        unrealized_return = (
            float((close * shares - position_cost) / position_cost)
            if position_cost > 0
            else 0.0
        )
        relative_return = float(equity / capital - benchmark_equity / capital)
        buy_budget = buy_capacity(
            cash, shares, close, order_notional, max_exposure
        )
        buy_capacity_ratio = float(buy_budget / equity) if equity > 0 else 0.0
        sellable_ratio = (
            float(D(sellable) * close / equity) if equity > 0 else 0.0
        )
        current_last_fill_side = str(state.get("last_fill_side") or "NONE")
        current_days_since_fill = days_since_fill(
            state.get("last_fill_index"), local_index
        )
        portfolio_observation = {
            "exposure": exposure,
            "cash_ratio": cash_ratio,
            "unrealized_return": unrealized_return,
            "drawdown": drawdown,
            "relative_return": relative_return,
            "buy_capacity_ratio": buy_capacity_ratio,
            "sellable_ratio": sellable_ratio,
            "last_fill_side": current_last_fill_side,
            "days_since_fill": current_days_since_fill,
            "action_advantage_cny": str(action_adv),
            "action_advantage_sign": action_adv_sign,
            "action_advantage_ratio": action_adv_ratio,
        }
        frame = portfolio_frame(
            frame,
            exposure=exposure,
            cash_ratio=cash_ratio,
            unrealized_return=unrealized_return,
            drawdown=drawdown,
            relative_return=relative_return,
            buy_capacity_ratio=buy_capacity_ratio,
            sellable_ratio=sellable_ratio,
            last_fill_side=current_last_fill_side,
            days_since_fill=current_days_since_fill,
            action_advantage_sign=action_adv_sign,
            action_advantage_ratio=action_adv_ratio,
            vetoed=execution["status"] == "VETO",
        )
        neural = controller.observe(frame, stimulus)
        decision = str(neural["side"])
        actions = Counter(state["action_counts"])
        actions[decision] += 1
        state["action_counts"] = dict(actions)
        state["cash"] = str(cash)
        state["shares"] = shares
        state["position_cost"] = str(position_cost)
        state["sellable_shares"] = sellable
        state["unsettled"] = unsettled
        state["anchor"] = str(equity)
        state["peak_equity"] = str(peak_equity)
        state["drawdown"] = drawdown
        state["action_advantage_ratio"] = action_adv_ratio
        state["action_advantage_sign"] = action_adv_sign
        state["last_portfolio_observation"] = portfolio_observation
        state["pending_signal"] = decision
        state["next_index"] = local_index + 1
        state["equity_curve"].append(float(equity))

        slot = (local_index + 1) % 2
        checkpoint = args.out / f"brain-{slot}.npz"
        controller.save(checkpoint)
        state["checkpoint"] = {"file": checkpoint.name, "sha256": sha256(checkpoint)}
        atomic_json(state_path, state)

        event = {
            "step": local_index + 1,
            "date": date,
            "symbol": SYMBOL,
            "open": str(row["open"]),
            "close": str(close),
            "dividend_per_share": str(dividend),
            "dividend_cash": str(dividend_cash),
            "execution": {
                **execution,
                "action_advantage_cny": str(action_adv),
                "action_advantage_sign": action_adv_sign,
                "action_advantage_ratio": action_adv_ratio,
            },
            "decision_for_next_open": decision,
            "cash_cny": str(cash),
            "shares": shares,
            "sellable_shares": sellable,
            "equity_cny": str(equity),
            "pnl_delta_cny": str(delta),
            "stimulus": stimulus,
            "feedback": {
                **feedback,
                "action_advantage_cny": str(action_adv),
                "action_advantage_sign": action_adv_sign,
                "action_advantage_ratio": action_adv_ratio,
            },
            "portfolio_observation": portfolio_observation,
            "neural": neural,
            "wall_seconds": time.monotonic() - started,
        }
        with events_path.open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        Image.fromarray(frame).save(args.out / "latest-input.png")
        atomic_json(args.out / "latest.json", event)
        print(
            json.dumps(
                {
                    "step": local_index + 1,
                    "date": date,
                    "decision": decision,
                    "execution": execution["status"],
                    "equity": str(equity),
                    "stimulus": stimulus,
                    "changed_edges": neural["memory"]["changed_edges"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        completed_now += 1

    if int(state["next_index"]) == len(rows):
        curve = [float(x) for x in state["equity_curve"]]
        final_equity = D(state["anchor"])
        summary = {
            "status": "complete",
            "symbol": SYMBOL,
            "start": str(rows[0]["date"]),
            "end": str(rows[-1]["date"]),
            "trading_days": len(rows),
            "initial_capital_cny": str(capital),
            "final_equity_cny": str(final_equity),
            "return": float(final_equity / capital - D("1")),
            "max_drawdown": max_drawdown(curve),
            "cash_cny": state["cash"],
            "shares": state["shares"],
            "sellable_shares": state["sellable_shares"],
            "unsettled": state["unsettled"],
            "position_cost_cny": state["position_cost"],
            "last_fill_side": state.get("last_fill_side"),
            "last_fill_index": state.get("last_fill_index"),
            "last_portfolio_observation": state.get("last_portfolio_observation", {}),
            "fees_cny": state["fees"],
            "buys": state["buy_count"],
            "sells": state["sell_count"],
            "vetoes": state["veto_count"],
            "veto_reasons": state.get("veto_reasons", {}),
            "reversal_count": state.get("reversal_count", 0),
            "reversal_veto_count": state.get("reversal_veto_count", 0),
            "action_advantage_cny": state.get("action_advantage_cny", "0"),
            "action_advantage_ratio": state.get("action_advantage_ratio", 0.0),
            "action_advantage_sign": state.get("action_advantage_sign", 0.0),
            "decisions": state["action_counts"],
            "buy_and_hold": baseline,
            "cash_return": 0.0,
            "paper_only": True,
            "source_data_sha256": prices_hash,
            "configuration": financial_config,
            "feedback_configuration": feedback_config,
            "learning": settings.learning,
            "initial_brain_sha256": initial_brain_hash,
            "signature": signature,
        }
        atomic_json(args.out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
