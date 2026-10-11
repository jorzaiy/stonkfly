#!/usr/bin/env python3
"""Multi-stock bear market stress test: 2018-2019 A-share rotation strategy.

Tests a simple equal-weight rotation strategy where Fly decides position sizing
for a basket of stocks. Uses DNp20 decoder for trading signals.

Stocks selected:
- 000001 平安银行 (Banking)
- 000002 万科A (Real Estate) 
- 000725 京东方A (Tech/Display)
- 000858 五粮液 (Consumer/Liquor)
- 002594 比亚迪 (Auto/EV)
- 300059 东方财富 (Finance/Tech)

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
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

from stonkfly.config import Settings
from stonkfly.display import market_frame
from stonkfly.neural.v51_controller import V51Controller


D = Decimal
LOT = 100
POLICY_VERSION = "bear-market-multi-stock-v1"


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


def create_basket_frame(
    symbols: List[str],
    price_histories: Dict[str, List[float]],
    positions: Dict[str, int],
    equity: float,
    cash: float,
    drawdown: float,
) -> np.ndarray:
    """Create a composite visual frame showing basket performance."""
    # Use the first stock's chart as base, overlay basket info
    if symbols and symbols[0] in price_histories:
        history = price_histories[symbols[0]]
        frame = market_frame(
            f"BASKET[{len(symbols)}]",
            history[-200:] if len(history) > 200 else history,
            D(str(history[-1])) * D("0.9995"),
            D(str(history[-1])) * D("1.0005"),
        )
    else:
        frame = np.zeros((64, 320, 3), dtype=np.uint8)
    
    # Add basket overlay
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 28, 319, 53), fill=(12, 22, 43))
    
    # Show basket metrics
    exposure = 1.0 - (cash / equity) if equity > 0 else 0.0
    cash_ratio = cash / equity if equity > 0 else 1.0
    
    metrics = [
        ("E", exposure, 0.0, 1.0),
        ("C", cash_ratio, 0.0, 1.0),
        ("D", drawdown, -0.25, 0.0),
        ("N", len([s for s, q in positions.items() if q > 0]), 0, len(symbols)),
    ]
    for index, (label, value, low, high) in enumerate(metrics):
        left = 3 + index * 79
        right = left + 73
        if isinstance(value, (int, float)):
            normalized = max(0.0, min(1.0, (value - low) / (high - low))) if high != low else 0.5
            fill = (44, 183, 116) if value >= 0 else (224, 73, 93)
            draw.text((left, 30), label, fill=(220, 230, 245))
            draw.rectangle((left + 10, 33, right, 46), outline=(94, 111, 143))
            width = round((right - left - 11) * normalized)
            if width:
                draw.rectangle((left + 11, 34, left + 10 + width, 45), fill=fill)
            if isinstance(value, float):
                draw.text((left + 10, 46), f"{value:+.1%}"[:7], fill=(180, 194, 220))
            else:
                draw.text((left + 10, 46), f"{value}", fill=(180, 194, 220))
    
    return np.asarray(image, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-stock bear market test 2018-2019")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with stock CSVs")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--start", default="2018-01-02", help="Start date")
    parser.add_argument("--end", default="2019-12-31", help="End date")
    parser.add_argument("--capital", default="100000", help="Initial capital in CNY")
    parser.add_argument("--order-pct", type=float, default=0.15, help="Order size as % of equity per stock")
    parser.add_argument("--commission-rate", default="0.0003", help="Commission rate")
    parser.add_argument("--minimum-commission", default="5", help="Minimum commission in CNY")
    parser.add_argument("--slippage", default="0.0005", help="Slippage per side")
    parser.add_argument("--stamp-duty", default="0.001", help="Stamp duty on sells")
    parser.add_argument("--max-exposure", default="0.85", help="Maximum total exposure")
    parser.add_argument("--decoder-type", default="dnp20", choices=["dnp20", "mbon"])
    parser.add_argument("--threshold", type=float, default=2.0, help="Decoder threshold Hz")
    parser.add_argument("--max-steps", type=int, default=0, help="Limit steps for testing")
    parser.add_argument("--rebalance-days", type=int, default=20, help="Days between rebalance")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    
    # Load all stock data
    stock_files = sorted(args.data_dir.glob("*.csv"))
    if not stock_files:
        raise SystemExit(f"No CSV files found in {args.data_dir}")
    
    stocks: Dict[str, List[dict]] = {}
    for f in stock_files:
        symbol = f.stem
        try:
            rows = load_prices(f)
            # Filter to date range
            rows = [r for r in rows if args.start <= str(r["date"]) <= args.end]
            if len(rows) >= 400:  # Need sufficient data
                stocks[symbol] = rows
                print(f"Loaded {symbol}: {len(rows)} days")
        except Exception as e:
            print(f"Skipping {symbol}: {e}")
    
    if len(stocks) < 3:
        raise SystemExit("Need at least 3 stocks with sufficient data")
    
    symbols = sorted(stocks.keys())
    print(f"\nUsing {len(symbols)} stocks: {symbols}")
    
    # Find common trading dates
    all_dates = set.intersection(*[{str(r["date"]) for r in rows} for rows in stocks.values()])
    common_dates = sorted(all_dates)
    print(f"Common trading days: {len(common_dates)}")
    
    # Build aligned price data
    aligned: Dict[str, Dict[str, dict]] = {s: {str(r["date"]): r for r in rows} for s, rows in stocks.items()}
    
    capital = D(args.capital)
    rate = D(args.commission_rate)
    minimum = D(args.minimum_commission)
    slippage = D(args.slippage)
    stamp_duty = D(args.stamp_duty)
    max_exposure = D(args.max_exposure)
    
    settings = Settings(
        learning=True,
        decoder_type=args.decoder_type,
        decoder_threshold_hz=args.threshold,
    )
    
    controller = V51Controller(settings)
    
    print(f"\nConfiguration:")
    print(f"  Decoder: {args.decoder_type}, threshold={args.threshold}Hz")
    print(f"  Capital: {capital} CNY")
    print(f"  Max exposure: {max_exposure}")
    print(f"  Rebalance every {args.rebalance_days} days")
    
    # State
    cash = capital
    positions: Dict[str, int] = {s: 0 for s in symbols}  # shares held
    sellable: Dict[str, int] = {s: 0 for s in symbols}   # T+1 sellable
    peak_equity = capital
    equity_curve: List[float] = []
    price_histories: Dict[str, List[float]] = {s: [] for s in symbols}
    
    events = []
    buy_count = 0
    sell_count = 0
    veto_count = 0
    action_counts: Dict[str, int] = Counter()
    reward_count = 0
    aversive_count = 0
    total_fees = D("0")
    
    state_path = args.out / "state.json"
    events_path = args.out / "events.jsonl"
    
    # Calculate buy-and-hold baseline (equal weight)
    bh_shares = {}
    bh_cash = capital
    first_day = common_dates[0]
    weight_per_stock = D("1") / len(symbols)
    for s in symbols:
        price = aligned[s][first_day]["open"] * (D("1") + slippage)
        budget = capital * weight_per_stock * max_exposure
        qty = int((budget / price / LOT).to_integral_value(rounding=ROUND_DOWN)) * LOT
        if qty > 0:
            cost = price * qty + commission(price * qty, rate, minimum)
            bh_cash -= cost
            bh_shares[s] = qty
    
    for day_idx, date in enumerate(common_dates):
        if args.max_steps and day_idx >= args.max_steps:
            break
        
        started = time.monotonic()
        
        # Update T+1 sellable
        for s in symbols:
            sellable[s] = positions[s]
        
        # Calculate current equity
        equity = cash
        for s in symbols:
            if s in aligned and date in aligned[s]:
                equity += aligned[s][date]["close"] * positions[s]
        
        # Update price histories
        for s in symbols:
            if s in aligned and date in aligned[s]:
                price_histories[s].append(float(aligned[s][date]["close"]))
        
        # Calculate drawdown
        peak_equity = max(peak_equity, equity)
        drawdown = float(equity / peak_equity - D("1"))
        
        # Create visual frame
        frame = create_basket_frame(
            symbols, price_histories, positions, float(equity), float(cash), drawdown
        )
        
        # Get neural decision
        neural = controller.observe(frame, "none")
        decision = str(neural["side"])
        action_counts[decision] += 1
        
        # Determine which stocks to trade based on momentum
        # Simple strategy: buy stocks with recent positive momentum, sell negative
        stock_signals: Dict[str, str] = {}
        for s in symbols:
            if len(price_histories[s]) >= 5:
                recent = price_histories[s][-5:]
                momentum = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0
                if momentum > 0.02:
                    stock_signals[s] = "BUY"
                elif momentum < -0.02:
                    stock_signals[s] = "SELL"
                else:
                    stock_signals[s] = "HOLD"
            else:
                stock_signals[s] = "HOLD"
        
        # Apply neural decision as a filter/multiplier
        executions = []
        day_turnover = D("0")
        
        if decision == "BUY":
            # Buy signals amplified
            for s in symbols:
                if stock_signals[s] == "BUY" and s in aligned and date in aligned[s]:
                    price = aligned[s][date]["open"] * (D("1") + slippage)
                    order_budget = equity * D(str(args.order_pct))
                    current_exposure = sum(
                        aligned[ss][date]["close"] * positions[ss] 
                        for ss in symbols if ss in aligned and date in aligned[ss]
                    ) / equity if equity > 0 else D("0")
                    
                    if current_exposure < max_exposure and cash > order_budget * D("0.5"):
                        qty = int((order_budget / price / LOT).to_integral_value(rounding=ROUND_DOWN)) * LOT
                        if qty > 0:
                            cost = price * qty
                            fee = commission(cost, rate, minimum)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                positions[s] += qty
                                total_fees += fee
                                day_turnover += cost
                                buy_count += 1
                                executions.append({"symbol": s, "side": "BUY", "qty": qty, "price": str(price)})
                            else:
                                veto_count += 1
                                
        elif decision == "SELL":
            # Sell signals amplified
            for s in symbols:
                if stock_signals[s] == "SELL" and sellable[s] > 0:
                    if s in aligned and date in aligned[s]:
                        price = aligned[s][date]["open"] * (D("1") - slippage)
                        qty = min(sellable[s], LOT * 10)  # Sell up to 1000 shares
                        if qty > 0:
                            proceeds = price * qty
                            fee = commission(proceeds, rate, minimum) + proceeds * stamp_duty
                            cash += proceeds - fee
                            positions[s] -= qty
                            sellable[s] -= qty
                            total_fees += fee
                            day_turnover += proceeds
                            sell_count += 1
                            executions.append({"symbol": s, "side": "SELL", "qty": qty, "price": str(price)})
        
        # Recalculate equity after trades
        equity = cash
        for s in symbols:
            if s in aligned and date in aligned[s]:
                equity += aligned[s][date]["close"] * positions[s]
        
        equity_curve.append(float(equity))
        
        # Simple feedback based on equity change
        if len(equity_curve) >= 2:
            ret = equity_curve[-1] / equity_curve[-2] - 1
            stimulus = "reward" if ret > 0.001 else "aversive" if ret < -0.001 else "none"
        else:
            stimulus = "none"
        
        if stimulus == "reward":
            controller.reinforce_linked("reward")
            reward_count += 1
        elif stimulus == "aversive":
            controller.reinforce_linked("aversive")
            aversive_count += 1
        
        # Log event
        event = {
            "day": day_idx + 1,
            "date": date,
            "decision": decision,
            "executions": executions,
            "equity": float(equity),
            "cash": float(cash),
            "positions": {s: positions[s] for s in symbols if positions[s] > 0},
            "drawdown": drawdown,
            "stimulus": stimulus,
            "turnover": float(day_turnover),
        }
        
        with events_path.open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        
        # Progress output
        if day_idx % 20 == 0 or executions:
            pos_count = len([s for s in symbols if positions[s] > 0])
            print(f"[{day_idx+1:3d}/{len(common_dates)}] {date} | {decision:4s} | "
                  f"equity={float(equity):,.0f} ({(float(equity)/float(capital)-1)*100:+.1f}%) | "
                  f"dd={drawdown*100:.1f}% | pos={pos_count}/{len(symbols)}")
    
    # Calculate buy-and-hold final equity
    last_day = common_dates[-1] if common_dates else first_day
    bh_equity = bh_cash
    for s in symbols:
        if s in aligned and last_day in aligned[s]:
            bh_equity += aligned[s][last_day]["close"] * bh_shares.get(s, 0)
    bh_return = float(bh_equity / capital - D("1"))
    
    # Calculate B&H drawdown
    bh_curve = []
    for date in common_dates:
        eq = bh_cash
        for s in symbols:
            if s in aligned and date in aligned[s]:
                eq += aligned[s][date]["close"] * bh_shares.get(s, 0)
        bh_curve.append(float(eq))
    bh_mdd = max_drawdown(bh_curve)
    
    # Final summary
    final_equity = equity_curve[-1] if equity_curve else float(capital)
    fly_return = final_equity / float(capital) - 1
    fly_mdd = max_drawdown(equity_curve) if equity_curve else 0
    
    summary = {
        "status": "complete",
        "stocks": symbols,
        "stock_count": len(symbols),
        "start": common_dates[0] if common_dates else args.start,
        "end": common_dates[-1] if common_dates else args.end,
        "trading_days": len(common_dates),
        "initial_capital_cny": str(capital),
        "final_equity_cny": f"{final_equity:.2f}",
        "return": fly_return,
        "max_drawdown": fly_mdd,
        "risk_reward_ratio": fly_return / abs(fly_mdd) if fly_mdd != 0 else 0,
        "fees_cny": str(total_fees),
        "buys": buy_count,
        "sells": sell_count,
        "vetoes": veto_count,
        "decisions": dict(action_counts),
        "reward_signals": reward_count,
        "aversive_signals": aversive_count,
        "buy_and_hold": {
            "final_equity_cny": f"{float(bh_equity):.2f}",
            "return": bh_return,
            "max_drawdown": bh_mdd,
        },
        "alpha_vs_bh": fly_return - bh_return,
        "drawdown_reduction": bh_mdd - fly_mdd,
        "paper_only": True,
        "decoder_type": args.decoder_type,
        "decoder_threshold": args.threshold,
    }
    atomic_json(args.out / "summary.json", summary)
    
    print("\n" + "="*70)
    print("MULTI-STOCK BEAR MARKET STRESS TEST COMPLETE")
    print("="*70)
    print(f"Stocks: {symbols}")
    print(f"Period: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} days)")
    print(f"\nFLY PERFORMANCE:")
    print(f"  Return:       {fly_return*100:+.2f}%")
    print(f"  Max Drawdown: {fly_mdd*100:.2f}%")
    print(f"  Risk/Reward:  {summary['risk_reward_ratio']:.3f}")
    print(f"\nBUY-AND-HOLD BASELINE (equal weight):")
    print(f"  Return:       {bh_return*100:+.2f}%")
    print(f"  Max Drawdown: {bh_mdd*100:.2f}%")
    print(f"\nALPHA:")
    print(f"  Return Alpha:     {summary['alpha_vs_bh']*100:+.2f}%")
    print(f"  DD Reduction:     {summary['drawdown_reduction']*100:+.2f}%")
    print(f"\nTRADING ACTIVITY:")
    print(f"  Buys: {buy_count}, Sells: {sell_count}, Vetoes: {veto_count}")
    print(f"  Decisions: {dict(action_counts)}")
    print(f"  Fees: {total_fees:.2f} CNY")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
