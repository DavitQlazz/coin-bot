#!/usr/bin/env python3
"""
Final High Frequency Strategy: Optimized Pair-Specific Parameters
Targeting 70+ trades with pair-specific ADX thresholds
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

def calculate_pair_specific_signals(df, pair, rsi_period=14, stoch_period=14, k_period=3, d_period=3,
                                   adx_period=14, overbought=75, oversold=25):
    """Calculate signals with pair-specific ADX thresholds"""
    try:
        # Pair-specific ADX thresholds
        pair_adx_thresholds = {
            'GBPCAD': 5,   # Low threshold for more trades
            'EURUSD': 5,   # Low threshold for more trades
            'USDCAD': 5,   # Low threshold for more trades
            'AUDUSD': 12,  # High threshold for quality (was poor performer)
            'EURJPY': 5,   # Low threshold for more trades
            'GBPUSD': 5    # Low threshold for more trades
        }

        adx_threshold = pair_adx_thresholds.get(pair, 5)  # Default to 5

        # Calculate Stochastic RSI
        stoch_rsi = StochRSIIndicator(close=df['Close'], window=rsi_period, smooth1=stoch_period, smooth2=k_period)
        df['stoch_rsi_k'] = stoch_rsi.stochrsi_k() * 100
        df['stoch_rsi_d'] = stoch_rsi.stochrsi_d() * 100

        # Calculate ADX for trend filter
        adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=adx_period)
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()

        # Generate signals with pair-specific ADX threshold
        df['signal'] = 0

        # Long signal
        long_condition = ((df['adx'] > adx_threshold) &  # Pair-specific threshold
                         (df['adx_pos'] > df['adx_neg']) &  # Uptrend
                         (df['stoch_rsi_k'] < oversold) & (df['stoch_rsi_d'] < oversold) &  # Relaxed levels
                         (df['stoch_rsi_k'] > df['stoch_rsi_d']) &  # K crossing above D
                         (df['stoch_rsi_k'].shift(1) <= df['stoch_rsi_d'].shift(1))  # Crossover
                        )
        df.loc[long_condition, 'signal'] = 1

        # Short signal
        short_condition = ((df['adx'] > adx_threshold) &  # Pair-specific threshold
                          (df['adx_neg'] > df['adx_pos']) &  # Downtrend
                          (df['stoch_rsi_k'] > overbought) & (df['stoch_rsi_d'] > overbought) &  # Relaxed levels
                          (df['stoch_rsi_k'] < df['stoch_rsi_d']) &  # K crossing below D
                          (df['stoch_rsi_k'].shift(1) >= df['stoch_rsi_d'].shift(1))  # Crossover
                         )
        df.loc[short_condition, 'signal'] = -1

        return df

    except Exception as e:
        print(f"Error calculating signals for {pair}: {str(e)}")
        return df

def run_final_high_freq_backtest(pairs, timeframe='1h', period_days=720,
                                rsi_period=14, stoch_period=14, k_period=3, d_period=3,
                                adx_period=14, overbought=75, oversold=25,
                                risk_per_trade=0.01, stop_loss_pct=0.015, take_profit_pct=0.03):
    """Run final high frequency backtest with pair-specific parameters targeting 70+ trades"""

    print("🎯 Final High Frequency Strategy: Pair-Specific Optimization")
    print("=" * 80)
    print("🚀 Hybrid Stochastic RSI + ADX - Pair-Specific Mode")
    print(f"Pairs: {', '.join(pairs)}")
    print(f"Timeframe: {timeframe}, Period: {period_days} days")
    print("Strategy: Pair-specific ADX thresholds - AUDUSD uses ADX=12 for quality")
    print(f"Stoch RSI: RSI={rsi_period}, Stoch={stoch_period}, K={k_period}, D={d_period}")
    print(f"Levels: Overbought={overbought}, Oversold={oversold} (RELAXED)")
    print(f"Risk: {risk_per_trade*100:.1f}% per trade, SL: {stop_loss_pct*100:.1f}%, TP: {take_profit_pct*100:.1f}%")
    print("=" * 80)

    all_trades = []
    pair_results = {}

    for pair in pairs:
        print(f"\n📊 Fetching data for {pair}...")
        df = fetch_forex_ohlcv(pair, timeframe, period_days)

        if df is None or df.empty:
            continue

        # Calculate pair-specific signals
        df = calculate_pair_specific_signals(df, pair, rsi_period, stoch_period, k_period, d_period,
                                           adx_period, overbought, oversold)

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
    print("📊 FINAL HIGH FREQUENCY PAIR-SPECIFIC RESULTS")
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

    if total_trades >= 70:
        print("🎉 TARGET ACHIEVED: 70+ trades!")
        print("✅ High frequency trading goal met!")
    elif total_trades >= 60:
        print("🎯 CLOSE TO TARGET: 60+ trades achieved!")
    else:
        print(f"📊 Results: {total_trades} trades")

    print("\n📋 PAIR-SPECIFIC ADX THRESHOLDS:")
    print("• GBPCAD/EURUSD/USDCAD/EURJPY/GBPUSD: ADX=5 (high volume)")
    print("• AUDUSD: ADX=12 (quality filter - was poor performer)")

    print("\n📋 TRADES PER PAIR:")
    for pair, results in pair_results.items():
        adx_threshold = 12 if pair == 'AUDUSD' else 5
        print(f"• {pair} (ADX={adx_threshold}): {results['trades']} trades ({results['win_rate']:.1f}% win rate)")

    # Save results
    results = {
        'strategy': 'Final High Frequency - Pair-Specific ADX',
        'parameters': {
            'rsi_period': rsi_period,
            'stoch_period': stoch_period,
            'k_period': k_period,
            'd_period': d_period,
            'adx_period': adx_period,
            'overbought': overbought,
            'oversold': oversold,
            'timeframe': timeframe,
            'period_days': period_days,
            'risk_per_trade': risk_per_trade,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'pair_specific_adx': {
                'GBPCAD': 5, 'EURUSD': 5, 'USDCAD': 5, 'AUDUSD': 12,
                'EURJPY': 5, 'GBPUSD': 5
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

    with open('final_high_freq_pair_specific_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Results saved to final_high_freq_pair_specific_results.json")

    return results

if __name__ == "__main__":
    # Include all pairs but with AUDUSD using stricter ADX threshold for quality
    all_pairs = ['GBPCAD', 'EURUSD', 'USDCAD', 'AUDUSD', 'EURJPY', 'GBPUSD']

    # Run the final strategy with pair-specific ADX thresholds
    results = run_final_high_freq_backtest(
        pairs=all_pairs,
        timeframe='1h',
        period_days=720,
        rsi_period=14,
        stoch_period=14,
        k_period=3,
        d_period=3,
        adx_period=14,
        overbought=75,  # Relaxed for volume
        oversold=25,    # Relaxed for volume
        risk_per_trade=0.01,  # Conservative risk
        stop_loss_pct=0.015,  # Tight stops
        take_profit_pct=0.03   # Small targets
    )

    print("\n✅ Final high frequency pair-specific strategy complete!")