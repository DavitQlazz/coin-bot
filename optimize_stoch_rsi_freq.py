#!/usr/bin/env python3
"""
Parameter Optimization for High-Frequency Stochastic RSI Trading
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

# Test Parameters
CRYPTO_EXCHANGE_ID = "binance"
TIMEFRAME = "4h"  # 4h timeframe for more trades over time
DAYS = 720  # 2 years for more data
START_BALANCE = 10000.0
POSITION_FRACTION = 0.015
STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 5.0
FETCH_LIMIT = 1000

def fetch_forex_ohlcv(symbol, period='720d', interval='1h'):
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
        print(f"Error fetching forex data: {e}")
        return pd.DataFrame()

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
        if len(bars) < FETCH_LIMIT:
            break
        since = bars[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
    if not all_bars:
        return pd.DataFrame()
    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("dt")
    return df

def calculate_stoch_rsi(df, k_period=14, d_period=3):
    """Calculate Stochastic RSI"""
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Calculate Stochastic of RSI
    stoch_rsi_k = ta.momentum.StochasticOscillator(rsi, rsi, rsi, window=k_period).stoch()
    stoch_rsi_d = ta.momentum.StochasticOscillator(rsi, rsi, rsi, window=k_period).stoch_signal()

    df['stoch_rsi_k'] = stoch_rsi_k
    df['stoch_rsi_d'] = stoch_rsi_d

    return df

def run_backtest(df, symbol, stoch_k, stoch_d, overbought, oversold):
    """Run backtest with given parameters"""
    trades = []
    equity = [{'timestamp': df.index[0].isoformat(), 'equity': START_BALANCE}]
    balance = START_BALANCE
    position = None
    pid = 0

    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        timestamp = df.index[i]

        # Calculate signals
        stoch_k_val = df['stoch_rsi_k'].iloc[i]
        stoch_d_val = df['stoch_rsi_d'].iloc[i]

        # Entry signals
        buy_signal = stoch_k_val < oversold and stoch_d_val < oversold
        sell_signal = stoch_k_val > overbought and stoch_d_val > overbought

        # Exit logic
        if position:
            entry_price = position['entry_price']
            if position['type'] == 'LONG':
                profit_pct = (current_price - entry_price) / entry_price * 100
                if profit_pct <= -STOP_LOSS_PCT or profit_pct >= TAKE_PROFIT_PCT:
                    pnl = (current_price - entry_price) * position['amount']
                    balance += pnl
                    trades.append({
                        'id': pid, 'symbol': symbol, 'strategy': 'Stoch RSI Opt',
                        'type': 'long', 'entry_ts': position['timestamp'].isoformat(),
                        'entry': entry_price, 'exit_ts': timestamp.isoformat(),
                        'exit': current_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
                    })
                    position = None
                    pid += 1
                    equity.append({'timestamp': timestamp.isoformat(), 'equity': balance})
            else:  # SHORT
                profit_pct = (entry_price - current_price) / entry_price * 100
                if profit_pct <= -STOP_LOSS_PCT or profit_pct >= TAKE_PROFIT_PCT:
                    pnl = (entry_price - current_price) * position['amount']
                    balance += pnl
                    trades.append({
                        'id': pid, 'symbol': symbol, 'strategy': 'Stoch RSI Opt',
                        'type': 'short', 'entry_ts': position['timestamp'].isoformat(),
                        'entry': entry_price, 'exit_ts': timestamp.isoformat(),
                        'exit': current_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
                    })
                    position = None
                    pid += 1
                    equity.append({'timestamp': timestamp.isoformat(), 'equity': balance})

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
        trades.append({
            'id': pid, 'symbol': symbol, 'strategy': 'Stoch RSI Opt',
            'type': position['type'].lower(), 'entry_ts': position['timestamp'].isoformat(),
            'entry': position['entry_price'], 'exit_ts': df.index[-1].isoformat(),
            'exit': final_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
        })
        equity.append({'timestamp': df.index[-1].isoformat(), 'equity': balance})

    summary = {
        'symbol': symbol,
        'trades': len(trades),
        'wins': sum(1 for t in trades if t['win'] == 1),
        'losses': sum(1 for t in trades if t['win'] == 0),
        'win_rate': (sum(1 for t in trades if t['win'] == 1) / len(trades) * 100) if trades else 0,
        'net_pnl': balance - START_BALANCE,
        'final_balance': balance,
        'parameters': {
            'stoch_k_period': stoch_k,
            'stoch_d_period': stoch_d,
            'overbought_level': overbought,
            'oversold_level': oversold,
            'timeframe': TIMEFRAME,
            'days': DAYS
        }
    }

    return trades, equity, summary

def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_stoch_rsi_freq.py <symbol>")
        return

    symbol = sys.argv[1]

    # Detect data source
    if symbol.endswith('=X'):
        data_source = 'forex'
    else:
        data_source = 'forex'
        symbol = symbol + '=X'

    print(f"🔬 Optimizing Stochastic RSI for High Frequency Trading")
    print(f"Symbol: {symbol}")
    print(f"Target: 80+ trades with 67%+ win rate")
    print(f"Timeframe: {TIMEFRAME}, Period: {DAYS} days")
    print("="*80)

    # Fetch data
    if data_source == 'crypto':
        exchange = ccxt.binance()
        now = datetime.now(UTC)
        since_dt = now - timedelta(days=DAYS)
        since_ms = int(since_dt.timestamp() * 1000)
        df = fetch_crypto_ohlcv(exchange, symbol, TIMEFRAME, since_ms)
    else:
        df = fetch_forex_ohlcv(symbol, f'{DAYS}d', TIMEFRAME)

    if df.empty:
        print('❌ No data fetched')
        return

    print(f"✅ Fetched {len(df)} candles")

    # Parameter combinations to test
    param_combinations = [
        # (stoch_k, stoch_d, overbought, oversold)
        (5, 3, 75, 25),   # Short period, tight levels
        (8, 3, 75, 25),   # Medium-short period
        (10, 3, 75, 25),  # Medium period
        (5, 3, 80, 20),   # Very tight levels
        (8, 3, 80, 20),   # Medium tight
        (3, 2, 75, 25),   # Very short periods
        (5, 3, 70, 30),   # Balanced
        (8, 3, 70, 30),   # Medium balanced
    ]

    best_result = None
    best_score = 0

    print(f"\n🧪 Testing {len(param_combinations)} parameter combinations...")

    for i, (stoch_k, stoch_d, overbought, oversold) in enumerate(param_combinations):
        print(f"\nTest {i+1}/{len(param_combinations)}: K={stoch_k}, D={stoch_d}, OB={overbought}, OS={oversold}")

        # Calculate indicators
        test_df = df.copy()
        test_df = calculate_stoch_rsi(test_df, stoch_k, stoch_d)

        # Run backtest
        trades, equity, summary = run_backtest(test_df, symbol, stoch_k, stoch_d, overbought, oversold)

        trades_count = summary['trades']
        win_rate = summary['win_rate']

        print(f"  Result: {trades_count} trades, {win_rate:.1f}% win rate, ${summary['net_pnl']:+.2f} P/L")

        # Score based on meeting targets
        if trades_count >= 80 and win_rate >= 67:
            score = 100 + (trades_count - 80) + (win_rate - 67)  # Bonus for exceeding targets
            result_type = "🎯 TARGET ACHIEVED!"
        elif trades_count >= 80:
            score = 50 + (trades_count - 80)  # Good trade count
            result_type = "📊 Good trade count"
        elif win_rate >= 67:
            score = 30 + (win_rate - 67)  # Good win rate
            result_type = "🎯 Good win rate"
        else:
            score = min(trades_count / 80 * 25, 25) + min(win_rate / 67 * 25, 25)  # Partial scores
            result_type = "📈 Partial success"

        print(f"  Score: {score:.1f} - {result_type}")

        if score > best_score:
            best_score = score
            best_result = {
                'summary': summary,
                'trades': trades,
                'equity': equity,
                'score': score
            }

    # Display best result
    print(f"\n" + "="*80)
    print(f"🏆 BEST RESULT (Score: {best_score:.1f})")
    print(f"="*80)

    if best_result:
        summary = best_result['summary']
        params = summary['parameters']

        print(f"Parameters:")
        print(f"  Stochastic K: {params['stoch_k_period']}")
        print(f"  Stochastic D: {params['stoch_d_period']}")
        print(f"  Overbought: {params['overbought_level']}")
        print(f"  Oversold: {params['oversold_level']}")
        print(f"  Timeframe: {params['timeframe']}")
        print(f"  Period: {params['days']} days")
        print(f"\nResults:")
        print(f"  Total Trades: {summary['trades']}")
        print(f"  Win Rate: {summary['win_rate']:.1f}%")
        print(f"  Net P/L: ${summary['net_pnl']:+.2f}")
        print(f"  Final Balance: ${summary['final_balance']:.2f}")

        if summary['trades'] >= 80 and summary['win_rate'] >= 67:
            print(f"\n🎉 SUCCESS! Target achieved: {summary['trades']} trades with {summary['win_rate']:.1f}% win rate")
        else:
            print(f"\n⚠️  Target not fully achieved. Best compromise found.")

        # Save best result
        safe_symbol = symbol.replace('/', '_').replace('=X', '')
        result_file = f'optimized_stoch_rsi_high_freq_{safe_symbol}.json'

        with open(result_file, 'w') as f:
            json.dump({
                'summary': summary,
                'trades': best_result['trades'],
                'equity': best_result['equity'],
                'optimization_score': best_score
            }, f, indent=2)

        print(f"\n💾 Best result saved to {result_file}")

    print(f"\n✅ Optimization complete!")

if __name__ == '__main__':
    main()