#!/usr/bin/env python3
"""
Bollinger Bands Mean Reversion Strategy: Alternative Approach
Classic mean reversion using Bollinger Bands
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
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

def calculate_bb_mean_reversion_signals(df, bb_period=20, bb_std=2,
                                       rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                                       adx_period=14, adx_threshold=25):
    """Calculate Bollinger Bands mean reversion signals"""
    try:
        # Calculate Bollinger Bands
        bb_indicator = BollingerBands(close=df['Close'], window=bb_period, window_dev=bb_std)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()

        # Calculate RSI for additional confirmation
        rsi_indicator = RSIIndicator(close=df['Close'], window=rsi_period)
        df['rsi'] = rsi_indicator.rsi()

        # Calculate ADX to avoid strong trends (mean reversion works better in ranging markets)
        adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=adx_period)
        df['adx'] = adx_indicator.adx()

        # Generate signals: Mean reversion when price touches bands
        df['signal'] = 0

        # Long signal: Price touches lower band + RSI oversold + weak trend (ADX low)
        long_condition = ((df['Low'] <= df['bb_lower']) &  # Price touches lower band
                         (df['rsi'] < rsi_oversold) &  # RSI oversold
                         (df['adx'] < adx_threshold)  # Weak trend (ranging market)
                        )
        df.loc[long_condition, 'signal'] = 1

        # Short signal: Price touches upper band + RSI overbought + weak trend (ADX low)
        short_condition = ((df['High'] >= df['bb_upper']) &  # Price touches upper band
                          (df['rsi'] > rsi_overbought) &  # RSI overbought
                          (df['adx'] < adx_threshold)  # Weak trend (ranging market)
                         )
        df.loc[short_condition, 'signal'] = -1

        return df

    except Exception as e:
        print(f"Error calculating BB signals: {str(e)}")
        return df

def run_bb_mean_reversion_backtest(pairs, timeframe='1h', period_days=720,
                                  bb_period=20, bb_std=2,
                                  rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                                  adx_period=14, adx_threshold=25,
                                  risk_per_trade=0.01, stop_loss_pct=0.015, take_profit_pct=0.025):
    """Run Bollinger Bands mean reversion backtest"""

    print("🎯 Bollinger Bands Mean Reversion Strategy: Alternative Approach")
    print("=" * 80)
    print("🚀 Bollinger Bands Mean Reversion + RSI + ADX Filter")
    print(f"Pairs: {', '.join(pairs)}")
    print(f"Timeframe: {timeframe}, Period: {period_days} days")
    print("Strategy: Mean reversion in ranging markets")
    print(f"BB: Period={bb_period}, Std Dev={bb_std}")
    print(f"RSI: Period={rsi_period}, OB={rsi_overbought}, OS={rsi_oversold}")
    print(f"ADX: Period={adx_period}, Threshold={adx_threshold} (Low for ranging)")
    print(f"Risk: {risk_per_trade*100:.1f}% per trade, SL: {stop_loss_pct*100:.1f}%, TP: {take_profit_pct*100:.1f}%")
    print("=" * 80)

    all_trades = []
    pair_results = {}

    for pair in pairs:
        print(f"\n📊 Fetching data for {pair}...")
        df = fetch_forex_ohlcv(pair, timeframe, period_days)

        if df is None or df.empty:
            continue

        # Calculate BB mean reversion signals
        df = calculate_bb_mean_reversion_signals(df, bb_period, bb_std,
                                                rsi_period, rsi_overbought, rsi_oversold,
                                                adx_period, adx_threshold)

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
    print("📊 BOLLINGER BANDS MEAN REVERSION RESULTS")
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

    if total_trades >= 25 and win_rate >= 55 and profit_factor > 1.5:
        print("🎉 EXCELLENT MEAN REVERSION: High win rate in ranging markets!")
    elif total_trades >= 20 and win_rate >= 50:
        print("🎯 SOLID MEAN REVERSION: Good performance!")
    elif win_rate >= 60:
        print("✅ STRONG MEAN REVERSION: Very high win rate!")
    else:
        print(f"📊 Results: {total_trades} trades, {win_rate:.1f}% win rate")

    print("\n📋 STRATEGY BREAKDOWN:")
    print("🎯 Bollinger Bands Mean Reversion Features:")
    print("• BB: 20-period, 2 standard deviations")
    print("• RSI: Oversold (<30) for longs, Overbought (>70) for shorts")
    print("• ADX: Low threshold (<25) for ranging markets")
    print("• Entry: Price touches band + RSI extreme + weak trend")

    print("\n📋 TRADES PER PAIR:")
    for pair, results in pair_results.items():
        print(f"• {pair}: {results['trades']} trades ({results['win_rate']:.1f}% win rate)")

    # Save results
    results = {
        'strategy': 'Bollinger Bands Mean Reversion',
        'parameters': {
            'bb_period': bb_period,
            'bb_std': bb_std,
            'rsi_period': rsi_period,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
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

    with open('bb_mean_reversion_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Results saved to bb_mean_reversion_results.json")

    return results

if __name__ == "__main__":
    # Test Bollinger Bands mean reversion on top performing pairs
    top_pairs = ['GBPCAD', 'USDCAD', 'EURJPY']

    # Run the BB mean reversion strategy
    results = run_bb_mean_reversion_backtest(
        pairs=top_pairs,
        timeframe='1h',
        period_days=720,
        bb_period=20,
        bb_std=2,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        adx_period=14,
        adx_threshold=25,  # Low ADX for ranging markets
        risk_per_trade=0.01,  # Lower risk for mean reversion
        stop_loss_pct=0.015,  # Tighter stops
        take_profit_pct=0.025  # Smaller targets
    )

    print("\n✅ Bollinger Bands mean reversion strategy complete!")