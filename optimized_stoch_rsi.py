#!/usr/bin/env python3
"""
Optimized Stochastic RSI Strategy for High Profitability
Enhanced version with better parameters and risk management
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

# Optimized Parameters
CRYPTO_EXCHANGE_ID = "binance"
TIMEFRAME = "1h"  # 1h timeframe
DAYS = 720  # 2 years maximum for 1h data
START_BALANCE = 10000.0
POSITION_FRACTION = 0.015  # Reduced risk per trade (1.5%)
STOP_LOSS_PCT = 2.0  # Updated stop loss
TAKE_PROFIT_PCT = 5.0  # Updated take profit
FETCH_LIMIT = 1000

# Stochastic RSI Parameters (very short periods for maximum signals)
STOCH_K_PERIOD = 3   # Very short for maximum signals
STOCH_D_PERIOD = 3
OVERBOUGHT_LEVEL = 70
OVERSOLD_LEVEL = 30


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


def fetch_forex_ohlcv(symbol, period='180d', interval='4h'):
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


def calculate_stoch_rsi(df, k_period=STOCH_K_PERIOD, d_period=STOCH_D_PERIOD):
    """Calculate Stochastic RSI"""
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Calculate Stochastic of RSI
    stoch_rsi_k = ta.momentum.StochasticOscillator(rsi, rsi, rsi, window=k_period).stoch()
    stoch_rsi_d = ta.momentum.StochasticOscillator(rsi, rsi, rsi, window=k_period).stoch_signal()

    df['stoch_rsi_k'] = stoch_rsi_k
    df['stoch_rsi_d'] = stoch_rsi_d

    return df


def generate_stoch_rsi_signal(df, i):
    """Generate Stochastic RSI trading signal"""
    if i < STOCH_K_PERIOD + 14:  # Need warmup for RSI + Stoch
        return 'HOLD'

    k = df['stoch_rsi_k'].iloc[i]
    d = df['stoch_rsi_d'].iloc[i]
    prev_k = df['stoch_rsi_k'].iloc[i-1]
    prev_d = df['stoch_rsi_d'].iloc[i-1]

    # Oversold condition: K and D below oversold level, and K crossing above D
    if k < OVERSOLD_LEVEL and d < OVERSOLD_LEVEL and prev_k <= prev_d and k > d:
        return 'BUY'

    # Overbought condition: K and D above overbought level, and K crossing below D
    elif k > OVERBOUGHT_LEVEL and d > OVERBOUGHT_LEVEL and prev_k >= prev_d and k < d:
        return 'SELL'

    return 'HOLD'


def run_optimized_stoch_rsi_backtest(df, symbol):
    """Run optimized Stochastic RSI backtest"""
    balance = START_BALANCE
    equity = [{'timestamp': df.index[0].isoformat(), 'equity': balance}]
    trades = []
    position = None
    pid = 0

    for i in range(STOCH_K_PERIOD + 20, len(df)):  # Start after full warmup
        current_price = df['close'].iloc[i]
        timestamp = df.index[i]

        # Generate signal
        signal = generate_stoch_rsi_signal(df, i)

        # Check for exit conditions if in position
        if position:
            profit_pct = ((current_price / position['entry_price']) - 1) * 100

            should_exit = False
            pnl = 0

            if position['type'] == 'LONG':
                if profit_pct <= -STOP_LOSS_PCT:  # Stop loss
                    pnl = (current_price - position['entry_price']) * position['amount']
                    should_exit = True
                elif profit_pct >= TAKE_PROFIT_PCT:  # Take profit
                    pnl = (current_price - position['entry_price']) * position['amount']
                    should_exit = True
            else:  # SHORT
                if profit_pct >= STOP_LOSS_PCT:  # Stop loss for shorts
                    pnl = (position['entry_price'] - current_price) * position['amount']
                    should_exit = True
                elif profit_pct <= -TAKE_PROFIT_PCT:  # Take profit for shorts
                    pnl = (position['entry_price'] - current_price) * position['amount']
                    should_exit = True

            if should_exit:
                balance += pnl
                trades.append({
                    'id': pid, 'symbol': symbol, 'strategy': 'Optimized Stochastic RSI',
                    'type': position['type'].lower(), 'entry_ts': position['timestamp'].isoformat(),
                    'entry': position['entry_price'], 'exit_ts': timestamp.isoformat(),
                    'exit': current_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
                })
                position = None
                pid += 1
                equity.append({'timestamp': timestamp.isoformat(), 'equity': balance})

        # Enter new position
        if signal == 'BUY' and not position:
            amount = (balance * POSITION_FRACTION) / current_price
            position = {
                'entry_price': current_price,
                'amount': amount,
                'timestamp': timestamp,
                'type': 'LONG'
            }
        elif signal == 'SELL' and not position:
            amount = (balance * POSITION_FRACTION) / current_price
            position = {
                'entry_price': current_price,
                'amount': amount,
                'timestamp': timestamp,
                'type': 'SHORT'
            }

    # Close any open position at end
    if position:
        final_price = df['close'].iloc[-1]
        if position['type'] == 'LONG':
            pnl = (final_price - position['entry_price']) * position['amount']
        else:
            pnl = (position['entry_price'] - final_price) * position['amount']
        balance += pnl
        trades.append({
            'id': pid, 'symbol': symbol, 'strategy': 'Optimized Stochastic RSI',
            'type': position['type'].lower(), 'entry_ts': position['timestamp'].isoformat(),
            'entry': position['entry_price'], 'exit_ts': df.index[-1].isoformat(),
            'exit': final_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
        })
        equity.append({'timestamp': df.index[-1].isoformat(), 'equity': balance})

    summary = {
        'strategy': 'Optimized Stochastic RSI',
        'symbol': symbol,
        'starting_balance': START_BALANCE,
        'ending_balance': balance,
        'trades': len(trades),
        'wins': sum(1 for t in trades if t['win'] == 1),
        'losses': sum(1 for t in trades if t['win'] == 0),
        'net_pnl': balance - START_BALANCE,
        'win_rate': (sum(1 for t in trades if t['win'] == 1) / len(trades) * 100) if trades else 0,
        'avg_win': np.mean([t['pnl'] for t in trades if t['win'] == 1]) if any(t['win'] == 1 for t in trades) else 0,
        'avg_loss': np.mean([t['pnl'] for t in trades if t['win'] == 0]) if any(t['win'] == 0 for t in trades) else 0,
        'profit_factor': (sum(t['pnl'] for t in trades if t['win'] == 1) / abs(sum(t['pnl'] for t in trades if t['win'] == 0))) if any(t['win'] == 0 for t in trades) and sum(t['pnl'] for t in trades if t['win'] == 1) > 0 else float('inf')
    }

    return trades, equity, summary


def plot_results(equity, symbol):
    """Plot equity curve"""
    times = [pd.to_datetime(x['timestamp']) for x in equity]
    vals = [x['equity'] for x in equity]

    plt.figure(figsize=(12, 6))
    plt.plot(times, vals, linewidth=2, color='blue')
    plt.axhline(START_BALANCE, linestyle='--', color='gray', alpha=0.7)
    plt.fill_between(times, START_BALANCE, vals,
                    where=(np.array(vals) >= START_BALANCE),
                    color='green', alpha=0.3)
    plt.fill_between(times, START_BALANCE, vals,
                    where=(np.array(vals) < START_BALANCE),
                    color='red', alpha=0.3)

    safe_symbol = symbol.replace('/', '_').replace('=X', '')
    plt.title(f'Optimized Stochastic RSI - {symbol} ({TIMEFRAME}, {DAYS} days)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fname = f'optimized_stoch_rsi_{safe_symbol}_{TIMEFRAME}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Equity curve saved to {fname}')
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python optimized_stoch_rsi.py <symbol>")
        print("Examples:")
        print("  python optimized_stoch_rsi.py EURUSD=X")
        print("  python optimized_stoch_rsi.py BTC/USDT")
        return

    symbol = sys.argv[1]

    # Detect data source
    if symbol.endswith('=X'):
        data_source = 'forex'
    elif '/' in symbol:
        data_source = 'crypto'
    else:
        data_source = 'forex'
        symbol = symbol + '=X'

    print(f"🚀 Optimized Stochastic RSI Strategy")
    print(f"Symbol: {symbol}")
    print(f"Data Source: {data_source.upper()}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Period: {DAYS} days")
    print(f"Risk per Trade: {POSITION_FRACTION*100}%")
    print(f"Stop Loss: {STOP_LOSS_PCT}%")
    print(f"Take Profit: {TAKE_PROFIT_PCT}%")
    print(f"Stoch RSI Levels: Oversold={OVERSOLD_LEVEL}, Overbought={OVERBOUGHT_LEVEL}")
    print("="*60)

    # Fetch data
    if data_source == 'crypto':
        exchange = getattr(ccxt, CRYPTO_EXCHANGE_ID)({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        try:
            exchange.load_markets()
        except Exception as e:
            print(f'Warning loading markets: {e}')
        now = datetime.now(UTC)
        since_dt = now - timedelta(days=DAYS)
        since_ms = int(since_dt.timestamp() * 1000)
        df = fetch_crypto_ohlcv(exchange, symbol, TIMEFRAME, since_ms)
    else:
        df = fetch_forex_ohlcv(symbol, f'{DAYS}d', TIMEFRAME)

    if df.empty:
        print('❌ No data fetched, exiting')
        return

    print(f"✅ Fetched {len(df)} candles")

    # Calculate indicators
    df = calculate_stoch_rsi(df)
    print("✅ Calculated Stochastic RSI indicators")

    # Run backtest
    print("🔄 Running optimized Stochastic RSI backtest...")
    trades, equity, summary = run_optimized_stoch_rsi_backtest(df, symbol)

    # Print results
    print("\n" + "="*60)
    print("📊 OPTIMIZED STOCHASTIC RSI RESULTS")
    print("="*60)
    print(f"Starting Balance: ${summary['starting_balance']:.2f}")
    print(f"Ending Balance:   ${summary['ending_balance']:.2f}")
    print(f"Net P/L:          ${summary['net_pnl']:+.2f}")
    print(f"Total Trades:     {summary['trades']}")
    print(f"Winning Trades:   {summary['wins']}")
    print(f"Losing Trades:    {summary['losses']}")
    print(f"Win Rate:         {summary['win_rate']:.1f}%")
    if summary['wins'] > 0:
        print(f"Average Win:      ${summary['avg_win']:.2f}")
    if summary['losses'] > 0:
        print(f"Average Loss:     ${summary['avg_loss']:.2f}")
    print(f"Profit Factor:    {summary['profit_factor']:.2f}")
    print("="*60)

    # Performance analysis
    if summary['trades'] > 0:
        if summary['net_pnl'] > 0:
            print("🎉 PROFITABLE STRATEGY!")
            if summary['win_rate'] > 60:
                print("   ⭐ Excellent win rate!")
            elif summary['profit_factor'] > 2:
                print("   ⭐ Strong profit factor!")
        else:
            print("⚠️  Strategy needs improvement")
            if summary['win_rate'] < 40:
                print("   📉 Low win rate - consider adjusting entry conditions")
            if summary['profit_factor'] < 1:
                print("   📉 Poor risk-reward - losses exceed wins")

    # Save results
    safe_symbol = symbol.replace('/', '_').replace('=X', '')
    results_file = f'optimized_stoch_rsi_results_{safe_symbol}_{TIMEFRAME}.json'

    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'trades': trades,
            'equity': equity,
            'parameters': {
                'symbol': symbol,
                'timeframe': TIMEFRAME,
                'days': DAYS,
                'position_fraction': POSITION_FRACTION,
                'stop_loss_pct': STOP_LOSS_PCT,
                'take_profit_pct': TAKE_PROFIT_PCT,
                'stoch_k_period': STOCH_K_PERIOD,
                'stoch_d_period': STOCH_D_PERIOD,
                'oversold_level': OVERSOLD_LEVEL,
                'overbought_level': OVERBOUGHT_LEVEL
            }
        }, f, indent=2)

    print(f"💾 Results saved to {results_file}")

    # Plot results
    try:
        plot_results(equity, symbol)
    except Exception as e:
        print(f"⚠️  Could not generate plot: {e}")

    print("\n✅ Backtest complete!")


if __name__ == '__main__':
    main()