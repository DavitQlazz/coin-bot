#!/usr/bin/env python3
"""
Multi-Pair Bollinger Bands Mean Reversion Strategy
Target: 80+ trades with 67%+ win rate
"""

import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, UTC
import sys
import ta
import time

# Strategy Parameters
TIMEFRAME = "4h"
DAYS = 720  # 2 years
START_BALANCE = 10000.0
POSITION_FRACTION = 0.015
STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 5.0
FETCH_LIMIT = 1000

# Bollinger Bands Parameters optimized for high win rate
BB_PERIOD = 10
BB_STD = 1.5  # Standard deviation

# Pairs to trade
PAIRS = ['EURUSD=X', 'GBPUSD=X', 'USDCAD=X', 'EURGBP=X', 'EURJPY=X', 'AUDUSD=X', 'GBPCAD=X']

def fetch_forex_ohlcv(symbol, period='720d', interval='4h'):
    """Fetch forex data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df.columns = df.columns.str.lower()

        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})

        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        return df
    except Exception as e:
        print(f"Error fetching forex data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_bollinger_signals(df, period=10, std=1.5):
    """Calculate Bollinger Bands signals with optimized parameters"""
    try:
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        df['bb_std'] = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (std * df['bb_std'])

        # Generate signals
        df['signal'] = 0

        # Buy signal: Price touches lower band (mean reversion)
        buy_signal = (df['low'] <= df['bb_lower']) & (df['low'].shift(1) > df['bb_lower'].shift(1))
        df.loc[buy_signal, 'signal'] = 1

        # Sell signal: Price touches upper band (mean reversion)
        sell_signal = (df['high'] >= df['bb_upper']) & (df['high'].shift(1) < df['bb_upper'].shift(1))
        df.loc[sell_signal, 'signal'] = -1

        return df

    except Exception as e:
        print(f"Error calculating Bollinger Bands signals: {str(e)}")
        return df

def run_multi_pair_bb_backtest():
    """Run Bollinger Bands mean reversion backtest across multiple pairs"""
    all_trades = []
    total_balance = START_BALANCE
    equity_curves = {pair: [] for pair in PAIRS}
    pair_balances = {pair: START_BALANCE / len(PAIRS) for pair in PAIRS}

    print(f"🚀 Multi-Pair Bollinger Bands Mean Reversion Strategy")
    print(f"Pairs: {', '.join(PAIRS)}")
    print(f"Timeframe: {TIMEFRAME}, Period: {DAYS} days")
    print(f"Target: 80+ trades with 67%+ win rate")
    print(f"Bollinger Bands: Period={BB_PERIOD}, Std={BB_STD}")
    print("="*80)

    # Fetch data for all pairs
    pair_data = {}
    for pair in PAIRS:
        print(f"📊 Fetching data for {pair}...")
        df = fetch_forex_ohlcv(pair, f'{DAYS}d', TIMEFRAME)
        if not df.empty:
            df = calculate_bollinger_signals(df)
            pair_data[pair] = df
            print(f"  ✅ {len(df)} candles")
            equity_curves[pair].append({'timestamp': df.index[0].isoformat(), 'equity': pair_balances[pair]})
        else:
            print(f"  ❌ No data for {pair}")
            pair_data[pair] = pd.DataFrame()

    # Run backtest for each pair
    for pair in PAIRS:
        if pair_data[pair].empty:
            continue

        df = pair_data[pair]
        balance = pair_balances[pair]
        position = None
        pid = len(all_trades)

        print(f"\n🔄 Backtesting {pair}...")

        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            timestamp = df.index[i]

            # Calculate Bollinger Band signals
            bb_upper = df['bb_upper'].iloc[i]
            bb_lower = df['bb_lower'].iloc[i]
            bb_middle = df['bb_middle'].iloc[i]

            # Entry signals - Mean reversion
            buy_signal = current_price < bb_lower  # Buy when price touches lower band
            sell_signal = current_price > bb_upper  # Sell when price touches upper band

            # Exit logic
            if position:
                entry_price = position['entry_price']
                if position['type'] == 'LONG':
                    profit_pct = (current_price - entry_price) / entry_price * 100
                    if profit_pct <= -STOP_LOSS_PCT or profit_pct >= TAKE_PROFIT_PCT:
                        pnl = (current_price - entry_price) * position['amount']
                        balance += pnl
                        all_trades.append({
                            'id': pid, 'symbol': pair, 'strategy': 'Multi-Pair BB',
                            'type': 'long', 'entry_ts': position['timestamp'].isoformat(),
                            'entry': entry_price, 'exit_ts': timestamp.isoformat(),
                            'exit': current_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
                        })
                        position = None
                        pid += 1
                        equity_curves[pair].append({'timestamp': timestamp.isoformat(), 'equity': balance})
                else:  # SHORT
                    profit_pct = (entry_price - current_price) / entry_price * 100
                    if profit_pct <= -STOP_LOSS_PCT or profit_pct >= TAKE_PROFIT_PCT:
                        pnl = (entry_price - current_price) * position['amount']
                        balance += pnl
                        all_trades.append({
                            'id': pid, 'symbol': pair, 'strategy': 'Multi-Pair BB',
                            'type': 'short', 'entry_ts': position['timestamp'].isoformat(),
                            'entry': entry_price, 'exit_ts': timestamp.isoformat(),
                            'exit': current_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
                        })
                        position = None
                        pid += 1
                        equity_curves[pair].append({'timestamp': timestamp.isoformat(), 'equity': balance})

            # Enter new position
            if buy_signal and not position:
                amount = (balance * POSITION_FRACTION) / current_price
                position = {
                    'entry_price': current_price,
                    'amount': amount,
                    'timestamp': timestamp,
                    'type': 'LONG'
                }
            elif sell_signal and not position:
                amount = (balance * POSITION_FRACTION) / current_price
                position = {
                    'entry_price': current_price,
                    'amount': amount,
                    'timestamp': timestamp,
                    'type': 'SHORT'
                }

        # Close any open position
        if position:
            final_price = df['close'].iloc[-1]
            if position['type'] == 'LONG':
                pnl = (final_price - position['entry_price']) * position['amount']
            else:
                pnl = (position['entry_price'] - final_price) * position['amount']
            balance += pnl
            all_trades.append({
                'id': pid, 'symbol': pair, 'strategy': 'Multi-Pair BB',
                'type': position['type'].lower(), 'entry_ts': position['timestamp'].isoformat(),
                'entry': position['entry_price'], 'exit_ts': df.index[-1].isoformat(),
                'exit': final_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
            })
            equity_curves[pair].append({'timestamp': df.index[-1].isoformat(), 'equity': balance})

        pair_balances[pair] = balance
        print(f"  ✅ {pair}: {len([t for t in all_trades if t['symbol'] == pair])} trades")

    # Calculate combined results
    total_trades = len(all_trades)
    winning_trades = sum(1 for t in all_trades if t['win'] == 1)
    losing_trades = sum(1 for t in all_trades if t['win'] == 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t['pnl'] for t in all_trades)
    final_balance = START_BALANCE + total_pnl

    # Create combined equity curve
    all_timestamps = set()
    for pair_trades in equity_curves.values():
        for point in pair_trades:
            all_timestamps.add(point['timestamp'])

    combined_equity = []
    for ts in sorted(all_timestamps):
        total_equity = 0
        for pair in PAIRS:
            pair_equity = equity_curves[pair]
            equity_val = START_BALANCE / len(PAIRS)
            for point in reversed(pair_equity):
                if point['timestamp'] <= ts:
                    equity_val = point['equity']
                    break
            total_equity += equity_val
        combined_equity.append({'timestamp': ts, 'equity': total_equity})

    summary = {
        'strategy': 'Multi-Pair Bollinger Bands Mean Reversion',
        'pairs': PAIRS,
        'timeframe': TIMEFRAME,
        'period_days': DAYS,
        'starting_balance': START_BALANCE,
        'ending_balance': final_balance,
        'net_pnl': total_pnl,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': np.mean([t['pnl'] for t in all_trades if t['win'] == 1]) if winning_trades > 0 else 0,
        'avg_loss': np.mean([t['pnl'] for t in all_trades if t['win'] == 0]) if losing_trades > 0 else 0,
        'profit_factor': (sum(t['pnl'] for t in all_trades if t['win'] == 1) / abs(sum(t['pnl'] for t in all_trades if t['win'] == 0))) if losing_trades > 0 and sum(t['pnl'] for t in all_trades if t['win'] == 1) > 0 else float('inf'),
        'parameters': {
            'bb_period': BB_PERIOD,
            'bb_std': BB_STD,
            'position_fraction': POSITION_FRACTION,
            'stop_loss_pct': STOP_LOSS_PCT,
            'take_profit_pct': TAKE_PROFIT_PCT
        }
    }

    return all_trades, combined_equity, summary

def main():
    print("🎯 Multi-Pair Bollinger Bands Mean Reversion Strategy")
    print("="*80)

    # Run multi-pair BB backtest
    trades, equity, summary = run_multi_pair_bb_backtest()

    # Display results
    print(f"\n" + "="*80)
    print(f"📊 MULTI-PAIR BOLLINGER BANDS RESULTS")
    print(f"="*80)
    print(f"Starting Balance: ${summary['starting_balance']:.2f}")
    print(f"Ending Balance:   ${summary['ending_balance']:.2f}")
    print(f"Net P/L:          ${summary['net_pnl']:+.2f}")
    print(f"Total Trades:     {summary['total_trades']}")
    print(f"Winning Trades:   {summary['winning_trades']}")
    print(f"Losing Trades:    {summary['losing_trades']}")
    print(f"Win Rate:         {summary['win_rate']:.1f}%")
    if summary['winning_trades'] > 0:
        print(f"Average Win:      ${summary['avg_win']:.2f}")
    if summary['losing_trades'] > 0:
        print(f"Average Loss:     ${summary['avg_loss']:.2f}")
    print(f"Profit Factor:    {summary['profit_factor']:.2f}")
    print(f"="*80)

    # Check if target achieved
    if summary['total_trades'] >= 80 and summary['win_rate'] >= 67:
        print("🎉 TARGET ACHIEVED! 80+ trades with 67%+ win rate!")
        print("🏆 Bollinger Bands strategy successful!")
    elif summary['total_trades'] >= 80:
        print(f"📊 Trade target achieved ({summary['total_trades']} trades) but win rate is {summary['win_rate']:.1f}% (target: 67%)")
    elif summary['win_rate'] >= 67:
        print(f"🎯 Win rate target achieved ({summary['win_rate']:.1f}%) but only {summary['total_trades']} trades (target: 80+)")
    else:
        print(f"⚠️  Target not achieved: {summary['total_trades']} trades, {summary['win_rate']:.1f}% win rate")

    # Trades per pair
    print(f"\n📋 TRADES PER PAIR:")
    for pair in PAIRS:
        pair_trades = [t for t in trades if t['symbol'] == pair]
        wins = sum(1 for t in pair_trades if t['win'] == 1)
        win_rate_pair = (wins / len(pair_trades) * 100) if pair_trades else 0
        print(f"• {pair}: {len(pair_trades)} trades ({win_rate_pair:.1f}% win rate)")

    # Save results
    result_file = 'multi_pair_bb_stoch_rsi.json'
    with open(result_file, 'w') as f:
        json.dump({
            'summary': summary,
            'trades': trades,
            'equity': equity
        }, f, indent=2)

    print(f"\n💾 Results saved to {result_file}")

    # Plot combined equity curve
    try:
        plot_multi_pair_equity(equity, summary)
    except Exception as e:
        print(f"⚠️  Could not generate plot: {e}")

    print(f"\n✅ Multi-pair Bollinger Bands backtest complete!")

def plot_multi_pair_equity(equity, summary):
    """Plot combined equity curve"""
    times = [pd.to_datetime(x['timestamp']) for x in equity]
    vals = [x['equity'] for x in equity]

    plt.figure(figsize=(14, 8))
    plt.plot(times, vals, linewidth=2, color='blue', label='Portfolio Equity')
    plt.axhline(summary['starting_balance'], linestyle='--', color='gray', alpha=0.7, label='Starting Balance')

    plt.fill_between(times, summary['starting_balance'], vals,
                    where=(np.array(vals) >= summary['starting_balance']),
                    color='green', alpha=0.3, label='Profitable')
    plt.fill_between(times, summary['starting_balance'], vals,
                    where=(np.array(vals) < summary['starting_balance']),
                    color='red', alpha=0.3, label='Losses')

    plt.title(f'Multi-Pair Bollinger Bands Mean Reversion\n{summary["total_trades"]} Trades, {summary["win_rate"]:.1f}% Win Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fname = 'multi_pair_bb_equity.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Equity curve saved to {fname}')
    plt.close()

if __name__ == '__main__':
    main()