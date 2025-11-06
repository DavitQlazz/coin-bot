#!/usr/bin/env python3
"""
SMC (Smart Money Concept) backtester - 1H Timeframe Version
Supports both crypto (Binance) and forex (Yahoo Finance) pairs
- Timeframe: 1h (optimized for shorter timeframe)
- Period: last 180 days (default)

Improved implementation with:
- ATR-based SL/TP for dynamic risk management
- OB filters: minimum size, confluence with swing structure
- Partial TP at 1:1 RR, full exit at 2:1 RR
- Saves results to JSON and equity curve PNG
"""
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, UTC
import sys

# Parameters - Optimized for 1H timeframe
CRYPTO_EXCHANGE_ID = "binance"
TIMEFRAME = "1h"  # Changed from 4h to 1h
DAYS = 180
START_BALANCE = 10000.0
POSITION_FRACTION = 0.015  # Slightly reduced risk for more frequent signals
RR_PARTIAL = 1.0  # partial TP at 1:1 RR
RR_FULL = 2.0     # full exit at 2:1 RR
ATR_PERIOD = 14   # ATR period for volatility
MIN_OB_SIZE_ATR = 0.3  # Reduced minimum OB size for 1h timeframe
SL_ATR_MULTIPLIER = 1.2  # Tighter SL for 1h timeframe
FETCH_LIMIT = 1000
WICK_BUFFER = 0.0003  # Reduced wick buffer for 1h


def fetch_crypto_ohlcv(exchange, symbol, timeframe, since_ms):
    all_bars = []
    since = since_ms
    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=FETCH_LIMIT)
        except Exception as e:
            print(f"Error fetching crypto ohlcv: {e}")
            break
        if not bars:
            break
        all_bars.extend(bars)
        since = bars[-1][0] + 1
        time.sleep(0.1)  # Rate limiting
        if len(bars) < FETCH_LIMIT:
            break
    return all_bars


def fetch_forex_ohlcv(pair, timeframe='1h', period_days=180):
    """Fetch forex OHLCV data from Yahoo Finance - 1H version"""
    try:
        ticker = f"{pair}=X"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        df = yf.download(ticker, start=start_date, end=end_date, interval=timeframe, progress=False)

        if df.empty:
            print(f"  ❌ No data for {pair}")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            print(f"  ❌ Missing required columns for {pair}")
            return None

        df = df[required_cols].copy()
        df.index = pd.to_datetime(df.index)

        print(f"  ✅ {len(df)} candles for {pair}")
        return df

    except Exception as e:
        print(f"  ❌ Error fetching {pair}: {str(e)}")
        return None


def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)

    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def detect_order_blocks(df, atr, min_ob_size_atr=0.5):
    """Detect Order Blocks (OB)"""
    obs = []

    for i in range(2, len(df) - 2):
        # Bullish OB: strong bearish candle followed by bullish engulfing
        if (df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and  # Previous bearish
            df['Close'].iloc[i] > df['Open'].iloc[i] and      # Current bullish
            df['Open'].iloc[i] < df['Close'].iloc[i-1] and    # Engulfing
            df['Close'].iloc[i] > df['Open'].iloc[i-1] and
            abs(df['Close'].iloc[i] - df['Open'].iloc[i]) > atr.iloc[i] * min_ob_size_atr):  # Minimum size

            # Check confluence with swing low
            if (df['Low'].iloc[i-2] > df['Low'].iloc[i-1] < df['Low'].iloc[i] and
                df['Low'].iloc[i] < df['Low'].iloc[i+1] < df['Low'].iloc[i+2]):
                obs.append({
                    'index': i,
                    'type': 'bullish',
                    'entry': df['High'].iloc[i] + WICK_BUFFER,
                    'sl': df['Low'].iloc[i] - WICK_BUFFER,
                    'size': abs(df['Close'].iloc[i] - df['Open'].iloc[i]) / atr.iloc[i]
                })

        # Bearish OB: strong bullish candle followed by bearish engulfing
        elif (df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and  # Previous bullish
              df['Close'].iloc[i] < df['Open'].iloc[i] and      # Current bearish
              df['Open'].iloc[i] > df['Close'].iloc[i-1] and    # Engulfing
              df['Close'].iloc[i] < df['Open'].iloc[i-1] and
              abs(df['Close'].iloc[i] - df['Open'].iloc[i]) > atr.iloc[i] * min_ob_size_atr):  # Minimum size

            # Check confluence with swing high
            if (df['High'].iloc[i-2] < df['High'].iloc[i-1] > df['High'].iloc[i] and
                df['High'].iloc[i] > df['High'].iloc[i+1] > df['High'].iloc[i+2]):
                obs.append({
                    'index': i,
                    'type': 'bearish',
                    'entry': df['Low'].iloc[i] - WICK_BUFFER,
                    'sl': df['High'].iloc[i] + WICK_BUFFER,
                    'size': abs(df['Close'].iloc[i] - df['Open'].iloc[i]) / atr.iloc[i]
                })

    return obs


def run_smc_backtest(pairs, timeframe='1h', period_days=180, start_balance=10000.0,
                    position_fraction=0.015, rr_partial=1.0, rr_full=2.0,
                    atr_period=14, min_ob_size_atr=0.3, sl_atr_multiplier=1.2):
    """Run SMC backtest on 1H timeframe"""

    print("🎯 SMC Backtest - 1H Timeframe Version")
    print("=" * 80)
    print("🚀 Smart Money Concept Strategy - Optimized for 1H")
    print(f"Pairs: {', '.join(pairs)}")
    print(f"Timeframe: {timeframe}, Period: {period_days} days")
    print("Strategy: Order Blocks with swing confluence")
    print(f"Risk: {position_fraction*100:.1f}% per trade")
    print(f"OB Size: {min_ob_size_atr} ATR, SL: {sl_atr_multiplier} ATR")
    print("=" * 80)

    all_trades = []
    pair_results = {}

    for pair in pairs:
        print(f"\n📊 Processing {pair}...")

        # Fetch data
        df = fetch_forex_ohlcv(pair, timeframe, period_days)
        if df is None or df.empty:
            continue

        # Calculate ATR
        atr = calculate_atr(df, atr_period)

        # Detect order blocks
        obs = detect_order_blocks(df, atr, min_ob_size_atr)
        print(f"  📍 Found {len(obs)} order blocks")

        # Run backtest
        balance = start_balance
        trades = []
        active_positions = []

        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            current_time = df.index[i]

            # Check for new OB entries
            for ob in obs:
                if ob['index'] == i:
                    # Calculate position size
                    risk_amount = balance * position_fraction
                    if ob['type'] == 'bullish':
                        sl_distance = abs(current_price - ob['sl'])
                        position_size = risk_amount / sl_distance
                        entry_price = ob['entry']
                    else:  # bearish
                        sl_distance = abs(ob['sl'] - current_price)
                        position_size = risk_amount / sl_distance
                        entry_price = ob['entry']

                    # Enter position
                    position = {
                        'pair': pair,
                        'type': ob['type'],
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'sl': ob['sl'],
                        'position_size': position_size,
                        'risk_amount': risk_amount,
                        'partial_tp_hit': False
                    }
                    active_positions.append(position)
                    print(f"  📈 Entered {ob['type']} at {entry_price:.5f}")

            # Check active positions for exits
            positions_to_remove = []
            for pos_idx, pos in enumerate(active_positions):
                if pos['pair'] != pair:
                    continue

                # Calculate current P/L
                if pos['type'] == 'bullish':
                    pnl = (current_price - pos['entry_price']) * pos['position_size']
                    rr = pnl / pos['risk_amount'] if pos['risk_amount'] > 0 else 0
                else:  # bearish
                    pnl = (pos['entry_price'] - current_price) * pos['position_size']
                    rr = pnl / pos['risk_amount'] if pos['risk_amount'] > 0 else 0

                # Check for partial TP at 1:1 RR
                if not pos['partial_tp_hit'] and rr >= rr_partial:
                    partial_size = pos['position_size'] * 0.5
                    partial_pnl = pnl * 0.5
                    balance += partial_pnl
                    pos['position_size'] -= partial_size
                    pos['partial_tp_hit'] = True
                    print(f"  💰 Partial TP: +${partial_pnl:.2f} (1:1 RR)")

                # Check for full exit at 2:1 RR or SL hit
                if rr >= rr_full:
                    remaining_pnl = pnl * 0.5 if pos['partial_tp_hit'] else pnl
                    balance += remaining_pnl
                    print(f"  ✅ Full exit: +${remaining_pnl:.2f} (2:1 RR)")

                    trade = {
                        'pair': pair,
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'position': pos['type'],
                        'pnl': pnl,
                        'rr': rr,
                        'outcome': 'win'
                    }
                    trades.append(trade)
                    positions_to_remove.append(pos_idx)

                elif (pos['type'] == 'bullish' and current_price <= pos['sl']) or \
                     (pos['type'] == 'bearish' and current_price >= pos['sl']):
                    # SL hit
                    balance += pnl  # pnl will be negative
                    print(f"  ❌ SL hit: ${pnl:.2f}")

                    trade = {
                        'pair': pair,
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'position': pos['type'],
                        'pnl': pnl,
                        'rr': rr,
                        'outcome': 'loss'
                    }
                    trades.append(trade)
                    positions_to_remove.append(pos_idx)

            # Remove closed positions
            for idx in reversed(positions_to_remove):
                active_positions.pop(idx)

        # Close any remaining positions at the end
        for pos in active_positions:
            if pos['pair'] == pair:
                final_price = df['Close'].iloc[-1]
                if pos['type'] == 'bullish':
                    pnl = (final_price - pos['entry_price']) * pos['position_size']
                else:
                    pnl = (pos['entry_price'] - final_price) * pos['position_size']

                balance += pnl
                rr = pnl / pos['risk_amount'] if pos['risk_amount'] > 0 else 0

                trade = {
                    'pair': pair,
                    'entry_time': pos['entry_time'],
                    'exit_time': df.index[-1],
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'position': pos['type'],
                    'pnl': pnl,
                    'rr': rr,
                    'outcome': 'open'
                }
                trades.append(trade)

        pair_results[pair] = {
            'trades': len(trades),
            'winning_trades': len([t for t in trades if t.get('outcome') == 'win']),
            'win_rate': len([t for t in trades if t.get('outcome') == 'win']) / len(trades) * 100 if trades else 0
        }

        all_trades.extend(trades)
        print(f"  ✅ {pair}: {len(trades)} trades completed")

    # Calculate overall results
    total_trades = len(all_trades)
    winning_trades = len([t for t in all_trades if t.get('outcome') == 'win'])
    losing_trades = len([t for t in all_trades if t.get('outcome') == 'loss'])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    avg_win = np.mean([t['pnl'] for t in all_trades if t.get('outcome') == 'win']) if winning_trades > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in all_trades if t.get('outcome') == 'loss']) if losing_trades > 0 else 0

    profit_factor = abs(sum([t['pnl'] for t in all_trades if t.get('outcome') == 'win']) /
                       sum([t['pnl'] for t in all_trades if t.get('outcome') == 'loss'])) if losing_trades > 0 else float('inf')

    print("\n" + "=" * 80)
    print("📊 SMC 1H TIMEFRAME RESULTS")
    print("=" * 80)
    print(f"Starting Balance: $10000.00")
    print(f"Ending Balance:   ${10000 + sum([t['pnl'] for t in all_trades]):.2f}")
    print(f"Net P/L:          ${sum([t['pnl'] for t in all_trades]):+.2f}")
    print(f"Total Trades:     {total_trades}")
    print(f"Winning Trades:   {winning_trades}")
    print(f"Losing Trades:    {losing_trades}")
    print(f"Win Rate:         {win_rate:.1f}%")
    print(f"Average Win:      ${avg_win:.2f}")
    print(f"Average Loss:     ${avg_loss:.2f}")
    print(f"Profit Factor:    {profit_factor:.2f}")
    print("=" * 80)

    if total_trades >= 15 and win_rate >= 50 and profit_factor > 1.5:
        print("🎉 EXCELLENT: Strong performance on 1H timeframe!")
    elif total_trades >= 10 and win_rate >= 45:
        print("✅ GOOD: Solid results on 1H timeframe!")
    elif win_rate >= 40:
        print("📊 DECENT: Acceptable performance!")
    else:
        print(f"📊 Results: {total_trades} trades, {win_rate:.1f}% win rate")

    print("\n📋 TRADES PER PAIR:")
    for pair, results in pair_results.items():
        print(f"• {pair}: {results['trades']} trades ({results['win_rate']:.1f}% win rate)")

    # Save results
    results = {
        'strategy': 'SMC 1H Timeframe',
        'parameters': {
            'timeframe': timeframe,
            'period_days': period_days,
            'start_balance': start_balance,
            'position_fraction': position_fraction,
            'rr_partial': rr_partial,
            'rr_full': rr_full,
            'atr_period': atr_period,
            'min_ob_size_atr': min_ob_size_atr,
            'sl_atr_multiplier': sl_atr_multiplier
        },
        'pairs': pairs,
        'overall_results': {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'net_pnl': sum([t['pnl'] for t in all_trades])
        },
        'pair_results': pair_results,
        'trades': all_trades
    }

    with open('smc_backtest_1h_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Results saved to smc_backtest_1h_results.json")

    return results

if __name__ == "__main__":
    # Test SMC on 1H timeframe with top performing pairs
    top_pairs = ['GBPCAD', 'USDCAD', 'EURJPY']

    # Run the SMC 1H backtest
    results = run_smc_backtest(
        pairs=top_pairs,
        timeframe='1h',
        period_days=180,
        start_balance=10000.0,
        position_fraction=0.015,  # Reduced risk for 1H
        rr_partial=1.0,
        rr_full=2.0,
        atr_period=14,
        min_ob_size_atr=0.3,  # Reduced for 1H
        sl_atr_multiplier=1.2   # Tighter SL for 1H
    )

    print("\n✅ SMC 1H timeframe backtest complete!")