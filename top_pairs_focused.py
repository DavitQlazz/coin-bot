#!/usr/bin/env python3
"""
Top Pairs Focused Strategy: GBPCAD, USDCAD, EURJPY
Optimized for best balance of trade frequency and win rate quality
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import StochRSIIndicator
from ta.trend import ADXIndicator
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

def fetch_forex_ohlcv(pair, timeframe='1h', period_days=720):
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

def calculate_optimized_signals(df, pair, rsi_period=14, stoch_period=14, k_period=3, d_period=3,
                               adx_period=14, overbought=75, oversold=25):
    """Calculate optimized signals for top performing pairs"""
    try:
        # Pair-specific optimizations based on previous performance
        pair_optimizations = {
            'GBPCAD': {'adx_threshold': 6, 'overbought': 78, 'oversold': 22},  # High win rate focus
            'USDCAD': {'adx_threshold': 6, 'overbought': 75, 'oversold': 25},  # Balanced approach
            'EURJPY': {'adx_threshold': 5, 'overbought': 75, 'oversold': 25}   # Volume focus
        }

        config = pair_optimizations.get(pair, {'adx_threshold': 6, 'overbought': 75, 'oversold': 25})
        adx_threshold = config['adx_threshold']
        overbought_level = config['overbought']
        oversold_level = config['oversold']

        # Calculate Stochastic RSI
        stoch_rsi = StochRSIIndicator(close=df['Close'], window=rsi_period, smooth1=stoch_period, smooth2=k_period)
        df['stoch_rsi_k'] = stoch_rsi.stochrsi_k() * 100
        df['stoch_rsi_d'] = stoch_rsi.stochrsi_d() * 100

        # Calculate ADX for trend filter
        adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=adx_period)
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()

        # Generate signals with pair-specific parameters
        df['signal'] = 0

        # Long signal
        long_condition = ((df['adx'] > adx_threshold) &  # Pair-specific threshold
                         (df['adx_pos'] > df['adx_neg']) &  # Uptrend
                         (df['stoch_rsi_k'] < oversold_level) & (df['stoch_rsi_d'] < oversold_level) &  # Pair-specific levels
                         (df['stoch_rsi_k'] > df['stoch_rsi_d']) &  # K crossing above D
                         (df['stoch_rsi_k'].shift(1) <= df['stoch_rsi_d'].shift(1))  # Crossover
                        )
        df.loc[long_condition, 'signal'] = 1

        # Short signal
        short_condition = ((df['adx'] > adx_threshold) &  # Pair-specific threshold
                          (df['adx_neg'] > df['adx_pos']) &  # Downtrend
                          (df['stoch_rsi_k'] > overbought_level) & (df['stoch_rsi_d'] > overbought_level) &  # Pair-specific levels
                          (df['stoch_rsi_k'] < df['stoch_rsi_d']) &  # K crossing below D
                          (df['stoch_rsi_k'].shift(1) >= df['stoch_rsi_d'].shift(1))  # Crossover
                         )
        df.loc[short_condition, 'signal'] = -1

        return df

    except Exception as e:
        print(f"Error calculating signals for {pair}: {str(e)}")
        return df

def run_top_pairs_focused_backtest(pairs, timeframe='1h', period_days=720,
                                  rsi_period=14, stoch_period=14, k_period=3, d_period=3,
                                  adx_period=14, risk_per_trade=0.012, stop_loss_pct=0.018, take_profit_pct=0.035):
    """Run focused backtest on top performing pairs for optimal balance"""

    print("🎯 Top Pairs Focused Strategy: GBPCAD, USDCAD, EURJPY")
    print("=" * 80)
    print("🚀 Hybrid Stochastic RSI + ADX - Top Pairs Focus")
    print(f"Pairs: {', '.join(pairs)}")
    print(f"Timeframe: {timeframe}, Period: {period_days} days")
    print("Strategy: Optimized parameters for each pair's strengths")
    print(f"Stoch RSI: RSI={rsi_period}, Stoch={stoch_period}, K={k_period}, D={d_period}")
    print(f"Risk: {risk_per_trade*100:.1f}% per trade, SL: {stop_loss_pct*100:.1f}%, TP: {take_profit_pct*100:.1f}%")
    print("=" * 80)

    print("\n📋 PAIR-SPECIFIC OPTIMIZATIONS:")
    print("• GBPCAD: ADX=6, Levels=78/22 (Quality Focus - 60% win rate)")
    print("• USDCAD: ADX=6, Levels=75/25 (Balanced - 50% win rate)")
    print("• EURJPY: ADX=5, Levels=75/25 (Volume Focus - 37% win rate)")
    print("=" * 80)

    all_trades = []
    pair_results = {}

    for pair in pairs:
        print(f"\n📊 Fetching data for {pair}...")
        df = fetch_forex_ohlcv(pair, timeframe, period_days)

        if df is None or df.empty:
            continue

        # Calculate optimized signals for this pair
        df = calculate_optimized_signals(df, pair, rsi_period, stoch_period, k_period, d_period, adx_period)

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
    print("📊 TOP PAIRS FOCUSED RESULTS")
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

    if total_trades >= 25 and win_rate >= 45:
        print("🎉 EXCELLENT BALANCE: Good trade count with high win rate!")
    elif total_trades >= 20 and win_rate >= 50:
        print("🎯 STRONG PERFORMANCE: Solid balance achieved!")
    elif win_rate >= 55:
        print("✅ QUALITY FOCUS: High win rate maintained!")
    else:
        print(f"📊 Results: {total_trades} trades, {win_rate:.1f}% win rate")

    print("\n📋 PERFORMANCE BREAKDOWN:")
    print("🎯 Top Performing Pairs (from previous testing):")
    print("• GBPCAD: Previously 60.0% win rate")
    print("• USDCAD: Previously 50.0% win rate")
    print("• EURJPY: Previously 36.8% win rate")

    print("\n📊 Current Optimized Results:")

    print("\n📋 TRADES PER PAIR:")
    for pair, results in pair_results.items():
        config = {
            'GBPCAD': 'ADX=6, Levels=78/22',
            'USDCAD': 'ADX=6, Levels=75/25',
            'EURJPY': 'ADX=5, Levels=75/25'
        }.get(pair, 'Standard')
        print(f"• {pair} ({config}): {results['trades']} trades ({results['win_rate']:.1f}% win rate)")

    # Save results
    results = {
        'strategy': 'Top Pairs Focused - Optimized Balance',
        'parameters': {
            'rsi_period': rsi_period,
            'stoch_period': stoch_period,
            'k_period': k_period,
            'd_period': d_period,
            'adx_period': adx_period,
            'timeframe': timeframe,
            'period_days': period_days,
            'risk_per_trade': risk_per_trade,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'pair_optimizations': {
                'GBPCAD': {'adx_threshold': 6, 'overbought': 78, 'oversold': 22},
                'USDCAD': {'adx_threshold': 6, 'overbought': 75, 'oversold': 25},
                'EURJPY': {'adx_threshold': 5, 'overbought': 75, 'oversold': 25}
            }
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

    with open('top_pairs_focused_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Results saved to top_pairs_focused_results.json")

    return results

if __name__ == "__main__":
    # Focus on the top performing pairs for best balance
    top_pairs = ['GBPCAD', 'USDCAD', 'EURJPY']

    # Run the focused strategy on top pairs
    results = run_top_pairs_focused_backtest(
        pairs=top_pairs,
        timeframe='1h',
        period_days=720,
        rsi_period=14,
        stoch_period=14,
        k_period=3,
        d_period=3,
        adx_period=14,
        risk_per_trade=0.012,  # Balanced risk
        stop_loss_pct=0.018,   # Reasonable stops
        take_profit_pct=0.035   # Realistic targets
    )

    print("\n✅ Top pairs focused strategy complete!")