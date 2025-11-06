#!/usr/bin/env python3
"""
Multi-Pair MACD Crossover Strategy Backtest
Target: 80+ trades with 67%+ win rate
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

def fetch_forex_ohlcv(pair, timeframe='4h', period_days=720):
    """Fetch forex OHLCV data from Yahoo Finance"""
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

        print(f"  ✅ {len(df)} candles")
        return df

    except Exception as e:
        print(f"  ❌ Error fetching {pair}: {str(e)}")
        return None

def calculate_macd_signals(df, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD signals"""
    try:
        # Calculate MACD
        macd_indicator = MACD(close=df['Close'], window_fast=fast_period, window_slow=slow_period, window_sign=signal_period)
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_diff'] = macd_indicator.macd_diff()

        # Generate signals
        df['signal'] = 0

        # Bullish crossover: MACD crosses above signal line
        bullish_crossover = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df.loc[bullish_crossover, 'signal'] = 1

        # Bearish crossover: MACD crosses below signal line
        bearish_crossover = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        df.loc[bearish_crossover, 'signal'] = -1

        return df

    except Exception as e:
        print(f"Error calculating MACD signals: {str(e)}")
        return df

def run_multi_pair_macd_backtest(pairs, timeframe='4h', period_days=720,
                                fast_period=12, slow_period=26, signal_period=9,
                                risk_per_trade=0.015, stop_loss_pct=0.02, take_profit_pct=0.05):
    """Run multi-pair MACD crossover backtest"""

    print("🎯 Multi-Pair MACD Crossover Strategy")
    print("=" * 80)
    print("🚀 Multi-Pair MACD Crossover Strategy")
    print(f"Pairs: {', '.join(pairs)}")
    print(f"Timeframe: {timeframe}, Period: {period_days} days")
    print("Target: 80+ trades with 67%+ win rate")
    print(f"MACD: Fast={fast_period}, Slow={slow_period}, Signal={signal_period}")
    print("=" * 80)

    all_trades = []
    pair_results = {}

    for pair in pairs:
        print(f"\n📊 Fetching data for {pair}...")
        df = fetch_forex_ohlcv(pair, timeframe, period_days)

        if df is None or df.empty:
            continue

        # Calculate MACD signals
        df = calculate_macd_signals(df, fast_period, slow_period, signal_period)

        # Run backtest for this pair
        print(f"\n🔄 Backtesting {pair}...")

        balance = 10000.0
        trades = []
        position = None
        entry_price = 0

        for i in range(len(df)):
            if df['signal'].iloc[i] == 0:
                continue

            current_price = df['Close'].iloc[i]
            current_time = df.index[i]

            # Check for exit conditions if we have a position
            if position is not None:
                price_change = (current_price - entry_price) / entry_price

                # Check stop loss or take profit
                if (position == 'long' and (price_change <= -stop_loss_pct or price_change >= take_profit_pct)) or \
                   (position == 'short' and (price_change >= stop_loss_pct or price_change <= -take_profit_pct)):

                    # Calculate P/L
                    if position == 'long':
                        pnl = (current_price - entry_price) / entry_price * balance * risk_per_trade
                    else:
                        pnl = (entry_price - current_price) / entry_price * balance * risk_per_trade

                    balance += pnl

                    trade = {
                        'pair': pair,
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': pnl / (balance - pnl) * 100 if balance - pnl > 0 else 0
                    }
                    trades.append(trade)
                    position = None

            # Enter new position
            if position is None and df['signal'].iloc[i] != 0:
                position = 'long' if df['signal'].iloc[i] == 1 else 'short'
                entry_price = current_price
                entry_time = current_time

        # Close any remaining position at the end
        if position is not None:
            final_price = df['Close'].iloc[-1]
            price_change = (final_price - entry_price) / entry_price

            if position == 'long':
                pnl = price_change * balance * risk_per_trade
            else:
                pnl = -price_change * balance * risk_per_trade

            balance += pnl

            trade = {
                'pair': pair,
                'entry_time': entry_time,
                'exit_time': df.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'position': position,
                'pnl': pnl,
                'pnl_pct': pnl / (balance - pnl) * 100 if balance - pnl > 0 else 0
            }
            trades.append(trade)

        pair_results[pair] = {
            'trades': len(trades),
            'winning_trades': len([t for t in trades if t['pnl'] > 0]),
            'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100 if trades else 0
        }

        all_trades.extend(trades)
        print(f"  ✅ {pair}: {len(trades)} trades")

    # Calculate overall results
    total_trades = len(all_trades)
    winning_trades = len([t for t in all_trades if t['pnl'] > 0])
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    avg_win = np.mean([t['pnl'] for t in all_trades if t['pnl'] > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in all_trades if t['pnl'] < 0]) if losing_trades > 0 else 0

    profit_factor = abs(sum([t['pnl'] for t in all_trades if t['pnl'] > 0]) /
                       sum([t['pnl'] for t in all_trades if t['pnl'] < 0])) if losing_trades > 0 else float('inf')

    print("\n" + "=" * 80)
    print("📊 MULTI-PAIR MACD RESULTS")
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

    if total_trades >= 80 and win_rate >= 67:
        print("🎉 TARGET ACHIEVED: 80+ trades with 67%+ win rate!")
    elif total_trades >= 80:
        print(f"📊 Trade target achieved ({total_trades} trades) but win rate is {win_rate:.1f}% (target: 67%)")
    else:
        print(f"📊 Need more trades. Current: {total_trades} (target: 80+)")

    print("\n📋 TRADES PER PAIR:")
    for pair, results in pair_results.items():
        print(f"• {pair}: {results['trades']} trades ({results['win_rate']:.1f}% win rate)")

    # Save results
    results = {
        'strategy': 'MACD Crossover',
        'parameters': {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'timeframe': timeframe,
            'period_days': period_days,
            'risk_per_trade': risk_per_trade,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        },
        'pairs': pairs,
        'overall_results': {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'net_pnl': sum([t['pnl'] for t in all_trades])
        },
        'pair_results': pair_results,
        'trades': all_trades
    }

    with open('multi_pair_macd_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Results saved to multi_pair_macd_results.json")

    return results

if __name__ == "__main__":
    # Forex pairs to test
    pairs = ['EURUSD', 'GBPUSD', 'USDCAD', 'EURGBP', 'EURJPY', 'AUDUSD', 'GBPCAD']

    # Run the backtest
    results = run_multi_pair_macd_backtest(
        pairs=pairs,
        timeframe='4h',
        period_days=720,
        fast_period=12,
        slow_period=26,
        signal_period=9,
        risk_per_trade=0.015,
        stop_loss_pct=0.02,
        take_profit_pct=0.05
    )

    print("\n✅ Multi-pair MACD backtest complete!")