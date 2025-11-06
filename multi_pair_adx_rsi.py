#!/usr/bin/env python3
"""
Multi-Pair ADX + RSI Trend-Following Strategy Backtest
Target: 80+ trades with 67%+ win rate
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator
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

def calculate_adx_rsi_signals(df, adx_period=14, rsi_period=14, adx_threshold=25,
                             rsi_overbought=70, rsi_oversold=30):
    """Calculate ADX + RSI trend-following signals"""
    try:
        # Calculate ADX
        adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=adx_period)
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()

        # Calculate RSI
        rsi_indicator = RSIIndicator(close=df['Close'], window=rsi_period)
        df['rsi'] = rsi_indicator.rsi()

        # Generate signals - only in trending markets (ADX > threshold)
        df['signal'] = 0

        # Long signal: ADX trending AND RSI oversold
        long_signal = (df['adx'] > adx_threshold) & (df['rsi'] < rsi_oversold) & (df['adx_pos'] > df['adx_neg'])
        df.loc[long_signal, 'signal'] = 1

        # Short signal: ADX trending AND RSI overbought
        short_signal = (df['adx'] > adx_threshold) & (df['rsi'] > rsi_overbought) & (df['adx_neg'] > df['adx_pos'])
        df.loc[short_signal, 'signal'] = -1

        return df

    except Exception as e:
        print(f"Error calculating ADX + RSI signals: {str(e)}")
        return df

def run_multi_pair_adx_rsi_backtest(pairs, timeframe='4h', period_days=720,
                                   adx_period=14, rsi_period=14, adx_threshold=25,
                                   rsi_overbought=70, rsi_oversold=30,
                                   risk_per_trade=0.015, stop_loss_pct=0.02, take_profit_pct=0.05):
    """Run multi-pair ADX + RSI trend-following backtest"""

    print("🎯 Multi-Pair ADX + RSI Trend-Following Strategy")
    print("=" * 80)
    print("🚀 Multi-Pair ADX + RSI Trend-Following Strategy")
    print(f"Pairs: {', '.join(pairs)}")
    print(f"Timeframe: {timeframe}, Period: {period_days} days")
    print("Target: 80+ trades with 67%+ win rate")
    print(f"ADX: Period={adx_period}, Threshold={adx_threshold}")
    print(f"RSI: Period={rsi_period}, Overbought={rsi_overbought}, Oversold={rsi_oversold}")
    print("=" * 80)

    all_trades = []
    pair_results = {}

    for pair in pairs:
        print(f"\n📊 Fetching data for {pair}...")
        df = fetch_forex_ohlcv(pair, timeframe, period_days)

        if df is None or df.empty:
            continue

        # Calculate ADX + RSI signals
        df = calculate_adx_rsi_signals(df, adx_period, rsi_period, adx_threshold, rsi_overbought, rsi_oversold)

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
    print("📊 MULTI-PAIR ADX + RSI RESULTS")
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
        'strategy': 'ADX + RSI Trend-Following',
        'parameters': {
            'adx_period': adx_period,
            'rsi_period': rsi_period,
            'adx_threshold': adx_threshold,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
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

    with open('multi_pair_adx_rsi_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Results saved to multi_pair_adx_rsi_results.json")

    return results

if __name__ == "__main__":
    # Forex pairs to test
    pairs = ['EURUSD', 'GBPUSD', 'USDCAD', 'EURGBP', 'EURJPY', 'AUDUSD', 'GBPCAD']

    # Run the backtest
    results = run_multi_pair_adx_rsi_backtest(
        pairs=pairs,
        timeframe='4h',
        period_days=720,
        adx_period=14,
        rsi_period=14,
        adx_threshold=25,
        rsi_overbought=70,
        rsi_oversold=30,
        risk_per_trade=0.015,
        stop_loss_pct=0.02,
        take_profit_pct=0.05
    )

    print("\n✅ Multi-pair ADX + RSI backtest complete!")