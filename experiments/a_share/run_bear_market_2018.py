#!/usr/bin/env python3
"""Bear market stress test: 2018-2019 A-share replay with V51 MBON decoder.

Tests the Fly's response to a severe bear market (2018: -31% drawdown) followed
by a recovery (2019: +75% rebound). Uses the validated MBON decoder configuration
(threshold=1.5, min_f=0.3, decay=900) from 2022 training.

This is a paper experiment for research purposes only.
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

import numpy as np
from PIL import Image, ImageDraw

from stonkfly.config import Settings
from stonkfly.display import market_frame
from stonkfly.neural.v51_controller import V51Controller


D = Decimal
LOT = 100
SYMBOL = "000001.SZ"  # Ping An Bank
POLICY_VERSION = "bear-market-stress-test-v1"


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
) -> dict[str, object]:
    """Turn relative return and risk changes into an engineered DAN stimulus."""
    portfolio_return = float(equity / previous_equity - D("1"))
    benchmark_return = float(
        benchmark_equity / previous_benchmark_equity - D("1")
    )
    excess_return = portfolio_return - benchmark_return
    drawdown_change = drawdown - previous_drawdown
    score = excess_return + drawdown_weight * drawdown_change
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
    }


def portfolio_frame(
    frame,
    *,
    exposure: float,
    cash_ratio: float,
    unrealized_return: float,
    drawdown: float,
    relative_return: float,
    vetoed: bool,
):
    """Add a fixed visual status strip; the neural graph still receives RGB only."""
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 28, 319, 53), fill=(12, 22, 43))
    metrics = [
        ("E", exposure, 0.0, 1.0),
        ("C", cash_ratio, 0.0, 1.0),
        ("P", unrealized_return, -0.25, 0.25),
        ("D", drawdown, -0.25, 0.0),
        ("A", relative_return, -0.15, 0.15),
    ]
    for index, (label, value, low, high) in enumerate(metrics):
        left = 3 + index * 63
        right = left + 57
        normalized = max(0.0, min(1.0, (value - low) / (high - low)))
        fill = (44, 183, 116) if value >= 0 else (224, 73, 93)
        draw.text((left, 30), label, fill=(220, 230, 245))
        draw.rectangle((left + 10, 33, right, 46), outline=(94, 111, 143))
        width = round((right - left - 11) * normalized)
        if width:
            draw.rectangle((left + 11, 34, left + 10 + width, 45), fill=fill)
        draw.text((left + 10, 46), f"{value:+.1%}"[:7], fill=(180, 194, 220))
    if vetoed:
        draw.rectangle((0, 28, 319, 53), outline=(255, 55, 55), width=2)
    return np.asarray(image, dtype=np.uint8)


def buy_and_hold(
    rows: list[dict[str, object]],
    capital: Decimal,
    rate: Decimal,
    minimum: Decimal,
    slippage: Decimal,
    stamp_duty: Decimal,
) -> tuple[dict[str, object], list[Decimal]]:
    """Baseline: buy at first open and hold to end."""
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
    for row in rows:
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
    parser = argparse.ArgumentParser(description="Bear market stress test 2018-2019")
    parser.add_argument("--prices", type=Path, required=True, help="CSV file with OHLCV data")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--start", default="2018-01-02", help="Start date")
    parser.add_argument("--end", default="2019-12-31", help="End date")
    parser.add_argument("--capital", default="100000", help="Initial capital in CNY")
    parser.add_argument("--order-notional", default="10000", help="Order size in CNY")
    parser.add_argument("--commission-rate", default="0.0003", help="Commission rate")
    parser.add_argument("--minimum-commission", default="5", help="Minimum commission in CNY")
    parser.add_argument("--slippage", default="0.0005", help="Slippage per side")
    parser.add_argument("--stamp-duty", default="0.001", help="Stamp duty on sells")
    parser.add_argument("--reward-deadband-bps", type=float, default=1.0)
    parser.add_argument("--drawdown-weight", type=float, default=0.25)
    parser.add_argument("--turnover-weight", type=float, default=0.001)
    parser.add_argument("--veto-penalty-bps", type=float, default=2.0)
    parser.add_argument("--max-exposure", default="0.85", help="Maximum position exposure")
    # MBON decoder config - use validated best settings
    parser.add_argument("--decoder-type", default="mbon", choices=["dnp20", "mbon", "mbon_ratio"])
    parser.add_argument("--threshold", type=float, default=1.5, help="MBON decoder threshold Hz")
    # Note: plasticity params (min_f=0.3, decay=900) are hardcoded in rule.py
    parser.add_argument("--initial-brain", type=Path, help="Load pre-trained brain checkpoint")
    parser.add_argument("--weights-only", action="store_true", 
                        help="Only restore weights from checkpoint, keep dynamic state fresh")
    parser.add_argument("--freeze-learning", action="store_true", help="Disable plasticity")
    parser.add_argument("--max-steps", type=int, default=0, help="Limit steps for testing")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    prices_hash = sha256(args.prices)
    rows_all = load_prices(args.prices)
    
    eval_indices = [
        i for i, row in enumerate(rows_all)
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
    stamp_duty = D(args.stamp_duty)
    max_exposure = D(args.max_exposure)
    
    if not D("0") < max_exposure < D("1"):
        raise SystemExit("--max-exposure must be between zero and one")

    # Build settings with MBON decoder
    # Note: plasticity params (min_f=0.3, decay=900) are hardcoded in rule.py
    settings = Settings(
        learning=not args.freeze_learning,
        decoder_type=args.decoder_type,
        decoder_threshold_hz=args.threshold,
    )
    
    print(f"Configuration:")
    print(f"  Decoder: {args.decoder_type}, threshold={args.threshold}Hz")
    print(f"  Plasticity: min_f=0.3, decay=900s (hardcoded in rule.py)")
    print(f"  Learning: {settings.learning}")
    print(f"  Date range: {args.start} to {args.end} ({len(rows)} days)")
    print(f"  Capital: {capital} CNY")

    financial_config = {
        "capital_cny": str(capital),
        "order_notional_cny": str(order_notional),
        "commission_rate": str(rate),
        "minimum_commission_cny": str(minimum),
        "slippage_each_side": str(slippage),
        "stamp_duty": str(stamp_duty),
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
    }
    
    neural_config = {
        "decoder_type": args.decoder_type,
        "decoder_threshold_hz": args.threshold,
        "plasticity_minimum_fraction": 0.3,  # hardcoded in rule.py
        "plasticity_decay_seconds": 900.0,   # hardcoded in rule.py
        "learning": settings.learning,
        "seed": args.seed,
    }

    signature_payload = {
        "symbol": SYMBOL,
        "prices_sha256": prices_hash,
        "start": args.start,
        "end": args.end,
        "financial": financial_config,
        "feedback": feedback_config,
        "neural": neural_config,
        "initial_brain_sha256": sha256(args.initial_brain) if args.initial_brain else None,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()

    state_path = args.out / "state.json"
    events_path = args.out / "events.jsonl"
    
    controller = V51Controller(settings)
    baseline, baseline_curve = buy_and_hold(
        rows, capital, rate, minimum, slippage, stamp_duty
    )
    
    print(f"Buy-and-hold baseline: {baseline['return']*100:.2f}%, max_dd={baseline['max_drawdown']*100:.2f}%")

    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["signature"] != signature:
            raise SystemExit("run configuration or source data changed; use a new --out")
        checkpoint = args.out / state["checkpoint"]["file"]
        if sha256(checkpoint) != state["checkpoint"]["sha256"]:
            raise SystemExit("checkpoint hash mismatch")
        controller.restore(checkpoint)
    else:
        if args.initial_brain:
            if args.weights_only:
                controller.restore_weights_only(args.initial_brain)
            else:
                controller.restore(args.initial_brain)
        state = {
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
            "fees": "0",
            "buy_count": 0,
            "sell_count": 0,
            "veto_count": 0,
            "action_counts": {},
            "equity_curve": [],
            "checkpoint": None,
            "reward_count": 0,
            "aversive_count": 0,
        }
        atomic_json(args.out / "provenance.json", signature_payload)

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

        # T+1 settlement
        matured = [lot for lot in unsettled if int(lot["release_index"]) <= local_index]
        sellable += sum(int(lot["quantity"]) for lot in matured)
        unsettled = [lot for lot in unsettled if int(lot["release_index"]) > local_index]

        execution: dict[str, object] = {"requested": state["pending_signal"], "status": "HOLD"}
        turnover_notional = D("0")
        pending = str(state["pending_signal"])
        
        if pending == "BUY":
            price = row["open"] * (D("1") + slippage)
            budget = buy_capacity(cash, shares, row["open"], order_notional, max_exposure)
            quantity = int((budget / price / LOT).to_integral_value(rounding=ROUND_DOWN)) * LOT
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
                unsettled.append({"quantity": quantity, "release_index": local_index + 1})
                state["fees"] = str(D(state["fees"]) + fee)
                state["buy_count"] = int(state["buy_count"]) + 1
                execution = {
                    "requested": pending, "status": "FILLED",
                    "quantity": quantity, "price": str(price), "fee": str(fee),
                }
            else:
                state["veto_count"] = int(state["veto_count"]) + 1
                execution = {"requested": pending, "status": "VETO", "reason": "cash_lot_or_exposure"}
                
        elif pending == "SELL":
            price = row["open"] * (D("1") - slippage)
            quantity = min(
                sellable,
                int((order_notional / price / LOT).to_integral_value(rounding=ROUND_DOWN)) * LOT,
            )
            if quantity:
                shares_before = shares
                notional = price * quantity
                fee = commission(notional, rate, minimum) + notional * stamp_duty
                cash += notional - fee
                shares -= quantity
                sellable -= quantity
                position_cost -= position_cost * D(quantity) / D(shares_before)
                turnover_notional = notional
                state["fees"] = str(D(state["fees"]) + fee)
                state["sell_count"] = int(state["sell_count"]) + 1
                execution = {
                    "requested": pending, "status": "FILLED",
                    "quantity": quantity, "price": str(price), "fee": str(fee),
                }
            else:
                state["veto_count"] = int(state["veto_count"]) + 1
                execution = {"requested": pending, "status": "VETO", "reason": "t_plus_1_or_inventory"}

        close = row["close"]
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
            vetoed=execution["status"] == "VETO" and pending != "HOLD",
            deadband_bps=args.reward_deadband_bps,
            drawdown_weight=args.drawdown_weight,
            turnover_weight=args.turnover_weight,
            veto_penalty_bps=args.veto_penalty_bps,
        )
        stimulus = str(feedback["stimulus"])
        
        if stimulus == "reward":
            state["reward_count"] = int(state.get("reward_count", 0)) + 1
        elif stimulus == "aversive":
            state["aversive_count"] = int(state.get("aversive_count", 0)) + 1

        # Build visual frame
        history = history_prefix + [float(r["close"]) for r in rows[: local_index + 1]]
        frame = market_frame(
            SYMBOL, history,
            close * (D("1") - slippage),
            close * (D("1") + slippage),
        )
        exposure = float(close * shares / equity) if equity > 0 else 0.0
        cash_ratio = float(cash / equity) if equity > 0 else 1.0
        unrealized_return = (
            float((close * shares - position_cost) / position_cost)
            if position_cost > 0 else 0.0
        )
        relative_return = float(equity / capital - benchmark_equity / capital)
        frame = portfolio_frame(
            frame,
            exposure=exposure,
            cash_ratio=cash_ratio,
            unrealized_return=unrealized_return,
            drawdown=drawdown,
            relative_return=relative_return,
            vetoed=execution["status"] == "VETO",
        )

        # Neural observation and decision
        # V51Controller separates observation from reinforcement:
        # 1. observe() processes RGB input (always with reinforcement="none")
        # 2. reinforce_linked() delivers reward/aversive signals separately
        neural = controller.observe(frame, "none")
        decision = str(neural["side"])
        
        # Deliver reinforcement signal if needed
        reinforce_result = None
        if stimulus in ("reward", "aversive"):
            reinforce_result = controller.reinforce_linked(stimulus)
            neural["reinforce_result"] = reinforce_result
        
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
        state["pending_signal"] = decision
        state["next_index"] = local_index + 1
        state["equity_curve"].append(float(equity))

        # Checkpoint
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
            "execution": execution,
            "decision_for_next_open": decision,
            "cash_cny": str(cash),
            "shares": shares,
            "sellable_shares": sellable,
            "equity_cny": str(equity),
            "pnl_delta_cny": str(delta),
            "stimulus": stimulus,
            "feedback": feedback,
            "portfolio_observation": {
                "exposure": exposure,
                "cash_ratio": cash_ratio,
                "unrealized_return": unrealized_return,
                "drawdown": drawdown,
                "relative_return": relative_return,
            },
            "neural": neural,
            "wall_seconds": time.monotonic() - started,
        }
        with events_path.open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        
        # Progress output every 20 days or on significant events
        if local_index % 20 == 0 or execution["status"] == "FILLED":
            print(
                f"[{local_index+1:3d}/{len(rows)}] {date} | "
                f"decision={decision:4s} exec={execution['status']:5s} | "
                f"equity={float(equity):,.0f} ({float(equity/capital-1)*100:+.1f}%) | "
                f"dd={drawdown*100:.1f}% | stim={stimulus}"
            )
        completed_now += 1

    # Final summary
    if int(state["next_index"]) == len(rows):
        curve = [float(x) for x in state["equity_curve"]]
        final_equity = D(state["anchor"])
        fly_return = float(final_equity / capital - D("1"))
        fly_mdd = max_drawdown(curve)
        
        summary = {
            "status": "complete",
            "symbol": SYMBOL,
            "start": str(rows[0]["date"]),
            "end": str(rows[-1]["date"]),
            "trading_days": len(rows),
            "initial_capital_cny": str(capital),
            "final_equity_cny": str(final_equity),
            "return": fly_return,
            "max_drawdown": fly_mdd,
            "risk_reward_ratio": fly_return / abs(fly_mdd) if fly_mdd != 0 else 0,
            "cash_cny": state["cash"],
            "shares": state["shares"],
            "fees_cny": state["fees"],
            "buys": state["buy_count"],
            "sells": state["sell_count"],
            "vetoes": state["veto_count"],
            "decisions": state["action_counts"],
            "reward_signals": state.get("reward_count", 0),
            "aversive_signals": state.get("aversive_count", 0),
            "buy_and_hold": baseline,
            "alpha_vs_bh": fly_return - baseline["return"],
            "drawdown_reduction": baseline["max_drawdown"] - fly_mdd,
            "paper_only": True,
            "source_data_sha256": prices_hash,
            "configuration": financial_config,
            "feedback_configuration": feedback_config,
            "neural_configuration": neural_config,
        }
        atomic_json(args.out / "summary.json", summary)
        
        print("\n" + "="*60)
        print("BEAR MARKET STRESS TEST COMPLETE")
        print("="*60)
        print(f"Period: {rows[0]['date']} to {rows[-1]['date']} ({len(rows)} days)")
        print(f"\nFLY PERFORMANCE:")
        print(f"  Return:       {fly_return*100:+.2f}%")
        print(f"  Max Drawdown: {fly_mdd*100:.2f}%")
        print(f"  Risk/Reward:  {summary['risk_reward_ratio']:.3f}")
        print(f"\nBUY-AND-HOLD BASELINE:")
        print(f"  Return:       {baseline['return']*100:+.2f}%")
        print(f"  Max Drawdown: {baseline['max_drawdown']*100:.2f}%")
        print(f"\nALPHA:")
        print(f"  Return Alpha:     {summary['alpha_vs_bh']*100:+.2f}%")
        print(f"  DD Reduction:     {summary['drawdown_reduction']*100:+.2f}%")
        print(f"\nTRADING ACTIVITY:")
        print(f"  Buys: {state['buy_count']}, Sells: {state['sell_count']}, Vetoes: {state['veto_count']}")
        print(f"  Decisions: {state['action_counts']}")
        print(f"  Rewards: {state.get('reward_count', 0)}, Aversive: {state.get('aversive_count', 0)}")
        print("="*60)
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
