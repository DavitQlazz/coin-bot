#!/usr/bin/env python3
"""
SMC (Smart Money Concept) backtester with best practices
Supports both crypto (Binance) and forex (Yahoo Finance) pairs
- Timeframe: 4h (default)
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

# Parameters
CRYPTO_EXCHANGE_ID = "binance"
TIMEFRAME = "4h"
DAYS = 180
START_BALANCE = 10000.0
POSITION_FRACTION = 0.02  # fraction of capital risked per trade
RR_PARTIAL = 1.0  # partial TP at 1:1 RR
RR_FULL = 2.0     # full exit at 2:1 RR
ATR_PERIOD = 14   # ATR period for volatility
MIN_OB_SIZE_ATR = 0.5  # minimum OB body size as fraction of ATR
SL_ATR_MULTIPLIER = 1.5  # SL distance as ATR multiplier
FETCH_LIMIT = 1000
WICK_BUFFER = 0.0005


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
            print(f"No forex data fetched for {symbol}")
            return pd.DataFrame()
        
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        
        # Rename timestamp column
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        
        # Keep required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        # Set timestamp as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        print(f"Fetched {len(df)} forex candles for {symbol}")
        return df
        
    except Exception as e:
        print(f"Error fetching forex data: {e}")
        return pd.DataFrame()


def fetch_ohlcv_since(data_source, symbol, timeframe_or_period, since_ms_or_period):
    """Unified data fetching for crypto and forex"""
    if data_source == 'crypto':
        exchange = getattr(ccxt, CRYPTO_EXCHANGE_ID)({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        try:
            exchange.load_markets()
        except Exception as e:
            print(f'Warning loading crypto markets: {e}')
        return fetch_crypto_ohlcv(exchange, symbol, timeframe_or_period, since_ms_or_period)
    elif data_source == 'forex':
        return fetch_forex_ohlcv(symbol, timeframe_or_period, since_ms_or_period)
    else:
        raise ValueError("data_source must be 'crypto' or 'forex'")


def calculate_atr(df, period=ATR_PERIOD):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def detect_order_blocks(df):
    """Detect order blocks with filters for quality"""
    obs = []
    n = len(df)
    atr = calculate_atr(df)
    
    for i in range(ATR_PERIOD, n - 3):  # start after ATR warmup
        o = df.iloc[i]
        body_size = abs(o['close'] - o['open'])
        current_atr = atr.iloc[i]
        
        # minimum body size filter
        if body_size < MIN_OB_SIZE_ATR * current_atr:
            continue
        
        # bullish OB: bullish candle + lower low in next 3 bars + OB at swing low
        if o['close'] > o['open']:
            if df['low'].iloc[i+1:i+4].min() < o['low']:
                # check if OB is at relative low (lower than recent highs)
                prev_highs = df['high'].iloc[max(0, i-3):i]
                if prev_highs.empty or o['high'] >= prev_highs.max():
                    zone_low = min(o['open'], o['close'])
                    zone_high = max(o['open'], o['close'])
                    obs.append({'type': 'bull', 'bar_index': i, 'zone_low': zone_low, 'zone_high': zone_high, 'ts': df.index[i], 'atr': current_atr})
        
        # bearish OB: bearish candle + higher high in next 3 bars + OB at swing high
        elif o['close'] < o['open']:
            if df['high'].iloc[i+1:i+4].max() > o['high']:
                # check if OB is at relative high (higher than recent lows)
                prev_lows = df['low'].iloc[max(0, i-3):i]
                if prev_lows.empty or o['low'] <= prev_lows.min():
                    zone_low = min(o['open'], o['close'])
                    zone_high = max(o['open'], o['close'])
                    obs.append({'type': 'bear', 'bar_index': i, 'zone_low': zone_low, 'zone_high': zone_high, 'ts': df.index[i], 'atr': current_atr})
    
    return obs


def fetch_ohlcv_since(exchange, symbol, timeframe, since_ms):
    all_bars = []
    since = since_ms
    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=FETCH_LIMIT)
        except Exception as e:
            print(f"Error fetching ohlcv: {e}")
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


def calculate_atr(df, period=ATR_PERIOD):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def detect_order_blocks(df):
    """Detect order blocks with filters for quality"""
    obs = []
    n = len(df)
    atr = calculate_atr(df)
    
    for i in range(ATR_PERIOD, n - 3):  # start after ATR warmup
        o = df.iloc[i]
        body_size = abs(o['close'] - o['open'])
        current_atr = atr.iloc[i]
        
        # minimum body size filter
        if body_size < MIN_OB_SIZE_ATR * current_atr:
            continue
        
        # bullish OB: bullish candle + lower low in next 3 bars + OB at swing low
        if o['close'] > o['open']:
            if df['low'].iloc[i+1:i+4].min() < o['low']:
                # check if OB is at relative low (lower than recent highs)
                prev_highs = df['high'].iloc[max(0, i-3):i]
                if prev_highs.empty or o['high'] >= prev_highs.max():
                    zone_low = min(o['open'], o['close'])
                    zone_high = max(o['open'], o['close'])
                    obs.append({'type': 'bull', 'bar_index': i, 'zone_low': zone_low, 'zone_high': zone_high, 'ts': df.index[i], 'atr': current_atr})
        
        # bearish OB: bearish candle + higher high in next 3 bars + OB at swing high
        elif o['close'] < o['open']:
            if df['high'].iloc[i+1:i+4].max() > o['high']:
                # check if OB is at relative high (higher than recent lows)
                prev_lows = df['low'].iloc[max(0, i-3):i]
                if prev_lows.empty or o['low'] <= prev_lows.min():
                    zone_low = min(o['open'], o['close'])
                    zone_high = max(o['open'], o['close'])
                    obs.append({'type': 'bear', 'bar_index': i, 'zone_low': zone_low, 'zone_high': zone_high, 'ts': df.index[i], 'atr': current_atr})
    
    return obs


def run_backtest(df, obs, symbol):
    balance = START_BALANCE
    equity = [{'timestamp': df.index[0].isoformat(), 'equity': balance}]
    trades = []
    pid = 0

    for ob in obs:
        start_i = ob['bar_index'] + 1
        for j in range(start_i, len(df)):
            low = df['low'].iloc[j]
            high = df['high'].iloc[j]
            zone_low = ob['zone_low'] * (1 - WICK_BUFFER)
            zone_high = ob['zone_high'] * (1 + WICK_BUFFER)
            # bar intersects zone
            if not (low <= zone_high and high >= zone_low):
                continue

            entry_atr = ob['atr']  # ATR at OB formation
            partial_taken = False

            if ob['type'] == 'bull':
                entry = min(zone_high, high)
                sl = entry - SL_ATR_MULTIPLIER * entry_atr
                risk_per_unit = entry - sl
                if risk_per_unit <= 0:
                    break
                risk_amount = balance * POSITION_FRACTION
                size = risk_amount / risk_per_unit
                tp_partial = entry + RR_PARTIAL * risk_per_unit
                tp_full = entry + RR_FULL * risk_per_unit

                result = None
                for k in range(j, len(df)):
                    if df['low'].iloc[k] <= sl:
                        pnl = -risk_amount
                        result = {'exit_ts': df.index[k], 'exit_price': sl, 'pnl': pnl, 'win': False}
                        break
                    if not partial_taken and df['high'].iloc[k] >= tp_partial:
                        # take partial profit at 1:1 RR
                        partial_pnl = RR_PARTIAL * risk_amount * 0.5  # 50% position
                        balance += partial_pnl
                        size *= 0.5  # reduce position
                        partial_taken = True
                        continue
                    if df['high'].iloc[k] >= tp_full:
                        pnl = RR_FULL * risk_amount * (0.5 if partial_taken else 1.0)
                        result = {'exit_ts': df.index[k], 'exit_price': tp_full, 'pnl': pnl, 'win': True}
                        break
                if not result:
                    last_price = df['close'].iloc[-1]
                    pnl = (last_price - entry) * size
                    result = {'exit_ts': df.index[-1], 'exit_price': last_price, 'pnl': pnl, 'win': pnl > 0}

                balance += result['pnl']
                trades.append({'id': pid, 'symbol': symbol, 'type': 'long', 'ob_ts': ob['ts'].isoformat(), 'entry_ts': df.index[j].isoformat(), 'entry': entry, 'sl': sl, 'tp_partial': tp_partial, 'tp_full': tp_full, 'exit_ts': result['exit_ts'].isoformat(), 'exit': result['exit_price'], 'pnl': result['pnl'], 'win': result['win']})
                pid += 1
                equity.append({'timestamp': result['exit_ts'].isoformat(), 'equity': balance})
                break

            else:  # bear
                entry = max(zone_low, low)
                sl = entry + SL_ATR_MULTIPLIER * entry_atr
                risk_per_unit = sl - entry
                if risk_per_unit <= 0:
                    break
                risk_amount = balance * POSITION_FRACTION
                size = risk_amount / risk_per_unit
                tp_partial = entry - RR_PARTIAL * risk_per_unit
                tp_full = entry - RR_FULL * risk_per_unit

                result = None
                for k in range(j, len(df)):
                    if df['high'].iloc[k] >= sl:
                        pnl = -risk_amount
                        result = {'exit_ts': df.index[k], 'exit_price': sl, 'pnl': pnl, 'win': False}
                        break
                    if not partial_taken and df['low'].iloc[k] <= tp_partial:
                        # take partial profit at 1:1 RR
                        partial_pnl = RR_PARTIAL * risk_amount * 0.5
                        balance += partial_pnl
                        size *= 0.5
                        partial_taken = True
                        continue
                    if df['low'].iloc[k] <= tp_full:
                        pnl = RR_FULL * risk_amount * (0.5 if partial_taken else 1.0)
                        result = {'exit_ts': df.index[k], 'exit_price': tp_full, 'pnl': pnl, 'win': True}
                        break
                if not result:
                    last_price = df['close'].iloc[-1]
                    pnl = (entry - last_price) * size
                    result = {'exit_ts': df.index[-1], 'exit_price': last_price, 'pnl': pnl, 'win': pnl > 0}

                balance += result['pnl']
                trades.append({'id': pid, 'symbol': symbol, 'type': 'short', 'ob_ts': ob['ts'].isoformat(), 'entry_ts': df.index[j].isoformat(), 'entry': entry, 'sl': sl, 'tp_partial': tp_partial, 'tp_full': tp_full, 'exit_ts': result['exit_ts'].isoformat(), 'exit': result['exit_price'], 'pnl': result['pnl'], 'win': result['win']})
                pid += 1
                equity.append({'timestamp': result['exit_ts'].isoformat(), 'equity': balance})
                break

    summary = {'starting_balance': START_BALANCE, 'ending_balance': balance, 'trades': len(trades), 'wins': sum(1 for t in trades if t['win']), 'losses': sum(1 for t in trades if not t['win']), 'net_pnl': balance - START_BALANCE}
    return trades, equity, summary


def save_results(trades, equity, summary, symbol):
    out = {'summary': summary, 'trades': trades, 'equity': equity}
    # Create filename based on symbol
    safe_symbol = symbol.replace('/', '_').replace('=X', '')
    fname = f'smc_backtest_results_{safe_symbol}.json'
    with open(fname, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {fname}")


def plot_equity(equity, symbol):
    times = [pd.to_datetime(x['timestamp']) for x in equity]
    vals = [x['equity'] for x in equity]
    plt.figure(figsize=(10, 5))
    plt.plot(times, vals, linewidth=2)
    plt.axhline(START_BALANCE, linestyle='--', color='gray')
    safe_symbol = symbol.replace('/', '_').replace('=X', '')
    plt.title(f'SMC Backtest Equity Curve - {symbol} ({DAYS} days, {TIMEFRAME})')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(alpha=0.3)
    outname = f'smc_equity_curve_{safe_symbol}.png'
    plt.tight_layout()
    plt.savefig(outname, dpi=150)
    print(f'Equity curve saved to {outname}')


def main():
    symbol = 'BTC/USDT'  # default crypto
    data_source = 'crypto'
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        # Detect data source based on symbol format
        if symbol.endswith('=X'):
            data_source = 'forex'
        elif '/' in symbol:
            data_source = 'crypto'
        else:
            # Assume forex if no slash and not =X
            data_source = 'forex'
            symbol = symbol + '=X'
    
    print(f"Data source: {data_source.upper()}")
    print(f"Symbol: {symbol}")
    
    if data_source == 'crypto':
        exchange = getattr(ccxt, CRYPTO_EXCHANGE_ID)({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        try:
            exchange.load_markets()
        except Exception as e:
            print(f'Warning loading markets: {e}')
        
        now = datetime.now(UTC)
        since_dt = now - timedelta(days=DAYS)
        since_ms = int(since_dt.timestamp() * 1000)
        period_or_since = since_ms
        interval_or_timeframe = TIMEFRAME
    else:  # forex
        period_or_since = f'{DAYS}d'
        interval_or_timeframe = TIMEFRAME
    
    if data_source == 'crypto':
        print(f'Fetching {symbol} {TIMEFRAME} data for last {DAYS} days...')
        df = fetch_crypto_ohlcv(exchange, symbol, TIMEFRAME, since_ms)
    else:
        print(f'Fetching {symbol} {TIMEFRAME} data for last {DAYS} days...')
        df = fetch_forex_ohlcv(symbol, f'{DAYS}d', TIMEFRAME)
    if df.empty:
        print('No data fetched, exiting')
        return

    print(f'Bars fetched: {len(df)}')
    obs = detect_order_blocks(df)
    print(f'Order blocks detected: {len(obs)}')
    trades, equity, summary = run_backtest(df, obs, symbol)
    print('=== Summary ===')
    print(f"Starting balance: ${summary['starting_balance']:.2f}")
    print(f"Ending balance:   ${summary['ending_balance']:.2f}")
    print(f"Trades: {summary['trades']}")
    print(f"Wins: {summary['wins']}  Losses: {summary['losses']}")
    print(f"Net PnL: ${summary['net_pnl']:+.2f}")

    save_results(trades, equity, summary, symbol)
    try:
        plot_equity(equity, symbol)
    except Exception as e:
        print(f'Could not plot equity curve: {e}')


if __name__ == '__main__':
    main()
