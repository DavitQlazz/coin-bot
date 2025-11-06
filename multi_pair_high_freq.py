#!/usr/bin/env python3
"""
Multi-Pair High-Frequency Stochastic RSI Strategy
Target: 80+ trades with 67%+ win rate across multiple pairs
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

# Strategy Parameters for quality + frequency balance
TIMEFRAME = "4h"  # 4h timeframe for quality
DAYS = 720  # 2 years for more trades
START_BALANCE = 10000.0
POSITION_FRACTION = 0.015
STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 5.0
FETCH_LIMIT = 1000

# Quality Stochastic RSI Parameters
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
OVERBOUGHT_LEVEL = 85
OVERSOLD_LEVEL = 45
USE_TREND_FILTER = False  # Remove trend filter

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

def calculate_stoch_rsi(df, k_period=STOCH_K_PERIOD, d_period=STOCH_D_PERIOD):
    """Calculate Stochastic RSI and trend indicators"""
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Calculate Stochastic of RSI
    stoch_rsi_k = ta.momentum.StochasticOscillator(rsi, rsi, rsi, window=k_period).stoch()
    stoch_rsi_d = ta.momentum.StochasticOscillator(rsi, rsi, rsi, window=k_period).stoch_signal()

    df['stoch_rsi_k'] = stoch_rsi_k
    df['stoch_rsi_d'] = stoch_rsi_d

    # Add trend filter (50-period SMA)
    if USE_TREND_FILTER:
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['trend'] = (df['close'] > df['sma_50']).astype(int)  # 1 for uptrend, 0 for downtrend

    return df

def run_multi_pair_backtest():
    """Run backtest across multiple pairs"""
    all_trades = []
    total_balance = START_BALANCE
    equity_curves = {pair: [] for pair in PAIRS}
    pair_balances = {pair: START_BALANCE / len(PAIRS) for pair in PAIRS}  # Equal allocation

    print(f"🚀 Multi-Pair High-Frequency Stochastic RSI Strategy")
    print(f"Pairs: {', '.join(PAIRS)}")
    print(f"Timeframe: {TIMEFRAME}, Period: {DAYS} days")
    print(f"Target: 80+ trades with 67%+ win rate")
    print(f"Stoch RSI: K={STOCH_K_PERIOD}, D={STOCH_D_PERIOD}, OB={OVERBOUGHT_LEVEL}, OS={OVERSOLD_LEVEL}")
    print("="*80)

    # Fetch data for all pairs
    pair_data = {}
    for pair in PAIRS:
        print(f"📊 Fetching data for {pair}...")
        df = fetch_forex_ohlcv(pair, f'{DAYS}d', TIMEFRAME)
        if not df.empty:
            df = calculate_stoch_rsi(df)
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

            # Calculate signals
            stoch_k_val = df['stoch_rsi_k'].iloc[i]
            stoch_d_val = df['stoch_rsi_d'].iloc[i]

            # Entry signals with trend filter
            buy_signal = (stoch_k_val < OVERSOLD_LEVEL and stoch_d_val < OVERSOLD_LEVEL and
                         (not USE_TREND_FILTER or df['trend'].iloc[i] == 1))  # Buy in uptrend
            sell_signal = (stoch_k_val > OVERBOUGHT_LEVEL and stoch_d_val > OVERBOUGHT_LEVEL and
                          (not USE_TREND_FILTER or df['trend'].iloc[i] == 0))  # Sell in downtrend

            # Exit logic
            if position:
                entry_price = position['entry_price']
                if position['type'] == 'LONG':
                    profit_pct = (current_price - entry_price) / entry_price * 100
                    if profit_pct <= -STOP_LOSS_PCT or profit_pct >= TAKE_PROFIT_PCT:
                        pnl = (current_price - entry_price) * position['amount']
                        balance += pnl
                        all_trades.append({
                            'id': pid, 'symbol': pair, 'strategy': 'Multi-Pair Stoch RSI',
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
                            'id': pid, 'symbol': pair, 'strategy': 'Multi-Pair Stoch RSI',
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
                'id': pid, 'symbol': pair, 'strategy': 'Multi-Pair Stoch RSI',
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
            # Find the equity value for this timestamp (or use last known)
            pair_equity = equity_curves[pair]
            equity_val = START_BALANCE / len(PAIRS)  # Default
            for point in reversed(pair_equity):
                if point['timestamp'] <= ts:
                    equity_val = point['equity']
                    break
            total_equity += equity_val
        combined_equity.append({'timestamp': ts, 'equity': total_equity})

    summary = {
        'strategy': 'Multi-Pair High-Frequency Stochastic RSI',
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
            'stoch_k_period': STOCH_K_PERIOD,
            'stoch_d_period': STOCH_D_PERIOD,
            'overbought_level': OVERBOUGHT_LEVEL,
            'oversold_level': OVERSOLD_LEVEL,
            'position_fraction': POSITION_FRACTION,
            'stop_loss_pct': STOP_LOSS_PCT,
            'take_profit_pct': TAKE_PROFIT_PCT
        }
    }

    return all_trades, combined_equity, summary

def main():
    print("🎯 Multi-Pair High-Frequency Trading Strategy")
    print("="*80)

    # Run multi-pair backtest
    trades, equity, summary = run_multi_pair_backtest()

    # Display results
    print(f"\n" + "="*80)
    print(f"📊 MULTI-PAIR HIGH-FREQUENCY RESULTS")
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
        print("🏆 High-frequency strategy successful!")
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
    result_file = 'multi_pair_high_freq_stoch_rsi.json'
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

    print(f"\n✅ Multi-pair backtest complete!")

def plot_multi_pair_equity(equity, summary):
    """Plot combined equity curve"""
    times = [pd.to_datetime(x['timestamp']) for x in equity]
    vals = [x['equity'] for x in equity]

    plt.figure(figsize=(14, 8))
    plt.plot(times, vals, linewidth=2, color='blue', label='Portfolio Equity')
    plt.axhline(summary['starting_balance'], linestyle='--', color='gray', alpha=0.7, label='Starting Balance')

    # Fill areas
    plt.fill_between(times, summary['starting_balance'], vals,
                    where=(np.array(vals) >= summary['starting_balance']),
                    color='green', alpha=0.3, label='Profitable')
    plt.fill_between(times, summary['starting_balance'], vals,
                    where=(np.array(vals) < summary['starting_balance']),
                    color='red', alpha=0.3, label='Losses')

    plt.title(f'Multi-Pair High-Frequency Stochastic RSI\n{summary["total_trades"]} Trades, {summary["win_rate"]:.1f}% Win Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fname = 'multi_pair_high_freq_equity.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Equity curve saved to {fname}')
    plt.close()

if __name__ == '__main__':
    main()