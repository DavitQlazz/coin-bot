#!/usr/bin/env python3
"""
SMC (Smart Money Concept) + Pattern Recognition backtester
Combines Order Blocks with Chart Pattern confirmation
Supports both crypto (Binance) and forex (Yahoo Finance) pairs
- Timeframe: 4h (default)
- Period: last 180 days (default)

Enhanced implementation with:
- ATR-based SL/TP for dynamic risk management
- OB filters: minimum size, confluence with swing structure
- Pattern confirmation: Double Top/Bottom, Head & Shoulders, Triangles
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
from scipy.signal import find_peaks

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

# Pattern detection parameters
DOUBLE_TOP_BOTTOM_TOLERANCE = 0.02  # 2% tolerance for peak/valley detection
HEAD_SHOULDERS_TOLERANCE = 0.015    # 1.5% tolerance for H&S pattern
TRIANGLE_MIN_POINTS = 4             # minimum points for triangle detection
PATTERN_LOOKBACK = 50               # bars to look back for pattern detection


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


def detect_double_top_bottom(df, lookback=PATTERN_LOOKBACK):
    """Detect Double Top and Double Bottom patterns"""
    patterns = []
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # Find peaks (potential double tops) and valleys (potential double bottoms)
    peak_indices, _ = find_peaks(highs, distance=5, prominence=np.std(highs)*0.5)
    valley_indices, _ = find_peaks(-lows, distance=5, prominence=np.std(lows)*0.5)

    # Check for double tops
    for i in range(len(peak_indices) - 1):
        peak1_idx = peak_indices[i]
        peak2_idx = peak_indices[i + 1]

        if peak2_idx - peak1_idx > 20:  # Minimum separation
            continue

        peak1_price = highs[peak1_idx]
        peak2_price = highs[peak2_idx]

        # Check if peaks are within tolerance
        if abs(peak1_price - peak2_price) / peak1_price <= DOUBLE_TOP_BOTTOM_TOLERANCE:
            # Find the valley between peaks
            between_lows = lows[peak1_idx:peak2_idx]
            valley_idx = peak1_idx + np.argmin(between_lows)
            valley_price = lows[valley_idx]

            # Check if valley is significantly lower
            if valley_price < peak1_price * 0.95:  # At least 5% drop
                patterns.append({
                    'type': 'double_top',
                    'peak1_idx': peak1_idx,
                    'peak2_idx': peak2_idx,
                    'valley_idx': valley_idx,
                    'peak_price': (peak1_price + peak2_price) / 2,
                    'valley_price': valley_price,
                    'neckline': valley_price,
                    'breakout_level': valley_price,
                    'pattern_end_idx': peak2_idx
                })

    # Check for double bottoms
    for i in range(len(valley_indices) - 1):
        valley1_idx = valley_indices[i]
        valley2_idx = valley_indices[i + 1]

        if valley2_idx - valley1_idx > 20:  # Minimum separation
            continue

        valley1_price = lows[valley1_idx]
        valley2_price = lows[valley2_idx]

        # Check if valleys are within tolerance
        if abs(valley1_price - valley2_price) / valley1_price <= DOUBLE_TOP_BOTTOM_TOLERANCE:
            # Find the peak between valleys
            between_highs = highs[valley1_idx:valley2_idx]
            peak_idx = valley1_idx + np.argmax(between_highs)
            peak_price = highs[peak_idx]

            # Check if peak is significantly higher
            if peak_price > valley1_price * 1.05:  # At least 5% rise
                patterns.append({
                    'type': 'double_bottom',
                    'valley1_idx': valley1_idx,
                    'valley2_idx': valley2_idx,
                    'peak_idx': peak_idx,
                    'valley_price': (valley1_price + valley2_price) / 2,
                    'peak_price': peak_price,
                    'neckline': peak_price,
                    'breakout_level': peak_price,
                    'pattern_end_idx': valley2_idx
                })

    return patterns


def detect_head_shoulders(df, lookback=PATTERN_LOOKBACK):
    """Detect Head and Shoulders patterns"""
    patterns = []
    highs = df['high'].values
    lows = df['low'].values

    # Find peaks for potential H&S tops
    peak_indices, _ = find_peaks(highs, distance=3, prominence=np.std(highs)*0.3)

    # Check for head and shoulders pattern
    for i in range(len(peak_indices) - 2):
        left_shoulder_idx = peak_indices[i]
        head_idx = peak_indices[i + 1]
        right_shoulder_idx = peak_indices[i + 2]

        # Check temporal order and spacing
        if not (left_shoulder_idx < head_idx < right_shoulder_idx):
            continue
        if head_idx - left_shoulder_idx > 15 or right_shoulder_idx - head_idx > 15:
            continue

        left_shoulder = highs[left_shoulder_idx]
        head = highs[head_idx]
        right_shoulder = highs[right_shoulder_idx]

        # Head should be higher than both shoulders
        if not (head > left_shoulder and head > right_shoulder):
            continue

        # Shoulders should be roughly equal
        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
        if shoulder_diff > HEAD_SHOULDERS_TOLERANCE:
            continue

        # Find neckline (lowest low between shoulders)
        neckline_lows = lows[min(left_shoulder_idx, right_shoulder_idx):max(left_shoulder_idx, right_shoulder_idx)]
        neckline = np.min(neckline_lows)

        patterns.append({
            'type': 'head_shoulders',
            'left_shoulder_idx': left_shoulder_idx,
            'head_idx': head_idx,
            'right_shoulder_idx': right_shoulder_idx,
            'left_shoulder': left_shoulder,
            'head': head,
            'right_shoulder': right_shoulder,
            'neckline': neckline,
            'breakout_level': neckline,
            'pattern_end_idx': right_shoulder_idx
        })

    return patterns


def detect_triangle_patterns(df, lookback=PATTERN_LOOKBACK):
    """Detect Triangle patterns (ascending, descending, symmetrical)"""
    patterns = []
    highs = df['high'].values
    lows = df['low'].values

    # Look for converging trendlines
    for start_idx in range(len(df) - TRIANGLE_MIN_POINTS, len(df) - 10):
        end_idx = min(start_idx + lookback, len(df))

        if end_idx - start_idx < TRIANGLE_MIN_POINTS:
            continue

        # Get data segment
        segment_highs = highs[start_idx:end_idx]
        segment_lows = lows[start_idx:end_idx]

        # Find peaks and valleys in segment
        peak_indices, _ = find_peaks(segment_highs, distance=2)
        valley_indices, _ = find_peaks(-segment_lows, distance=2)

        if len(peak_indices) < 2 or len(valley_indices) < 2:
            continue

        # Adjust indices to global dataframe
        peak_indices = peak_indices + start_idx
        valley_indices = valley_indices + start_idx

        # Check for triangle patterns
        # Symmetrical triangle: both trendlines converging
        if len(peak_indices) >= 2 and len(valley_indices) >= 2:
            # Fit trendlines
            peak_prices = highs[peak_indices]
            valley_prices = lows[valley_indices]

            # Upper trendline (peaks)
            if len(peak_indices) >= 2:
                upper_slope = np.polyfit(range(len(peak_indices)), peak_prices, 1)[0]

            # Lower trendline (valleys)
            if len(valley_indices) >= 2:
                lower_slope = np.polyfit(range(len(valley_indices)), valley_prices, 1)[0]

            # Check convergence (slopes should have opposite signs and be converging)
            if upper_slope < 0 and lower_slope > 0:  # Converging
                triangle_height = np.mean(peak_prices) - np.mean(valley_prices)
                if triangle_height > 0:
                    patterns.append({
                        'type': 'symmetrical_triangle',
                        'start_idx': start_idx,
                        'end_idx': end_idx - 1,
                        'upper_trendline_slope': upper_slope,
                        'lower_trendline_slope': lower_slope,
                        'height': triangle_height,
                        'breakout_level': (np.mean(peak_prices) + np.mean(valley_prices)) / 2,
                        'pattern_end_idx': end_idx - 1
                    })

        # Ascending triangle: horizontal resistance, rising support
        elif len(peak_indices) >= 2 and len(valley_indices) >= 2:
            peak_prices = highs[peak_indices]
            valley_prices = lows[valley_indices]

            # Check if peaks are roughly horizontal
            peak_variation = np.std(peak_prices) / np.mean(peak_prices)
            if peak_variation < 0.02:  # Less than 2% variation
                # Check if valleys are rising
                valley_trend = np.polyfit(range(len(valley_indices)), valley_prices, 1)[0]
                if valley_trend > 0:
                    patterns.append({
                        'type': 'ascending_triangle',
                        'start_idx': start_idx,
                        'end_idx': end_idx - 1,
                        'resistance_level': np.mean(peak_prices),
                        'support_slope': valley_trend,
                        'height': np.mean(peak_prices) - np.mean(valley_prices),
                        'breakout_level': np.mean(peak_prices),
                        'pattern_end_idx': end_idx - 1
                    })

        # Descending triangle: horizontal support, falling resistance
        elif len(peak_indices) >= 2 and len(valley_indices) >= 2:
            peak_prices = highs[peak_indices]
            valley_prices = lows[valley_indices]

            # Check if valleys are roughly horizontal
            valley_variation = np.std(valley_prices) / np.mean(valley_prices)
            if valley_variation < 0.02:  # Less than 2% variation
                # Check if peaks are falling
                peak_trend = np.polyfit(range(len(peak_indices)), peak_prices, 1)[0]
                if peak_trend < 0:
                    patterns.append({
                        'type': 'descending_triangle',
                        'start_idx': start_idx,
                        'end_idx': end_idx - 1,
                        'support_level': np.mean(valley_prices),
                        'resistance_slope': peak_trend,
                        'height': np.mean(peak_prices) - np.mean(valley_prices),
                        'breakout_level': np.mean(valley_prices),
                        'pattern_end_idx': end_idx - 1
                    })

    return patterns


def detect_chart_patterns(df):
    """Detect all chart patterns"""
    patterns = []

    # Detect double tops/bottoms
    double_patterns = detect_double_top_bottom(df)
    patterns.extend(double_patterns)

    # Detect head and shoulders
    hs_patterns = detect_head_shoulders(df)
    patterns.extend(hs_patterns)

    # Detect triangle patterns
    triangle_patterns = detect_triangle_patterns(df)
    patterns.extend(triangle_patterns)

    return patterns


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


def find_pattern_confirmation(ob, patterns, df):
    """Check if order block has pattern confirmation"""
    ob_idx = ob['bar_index']
    ob_type = ob['type']

    # Look for patterns that ended recently (within last 20 bars)
    recent_patterns = [p for p in patterns if abs(p['pattern_end_idx'] - ob_idx) <= 20]

    for pattern in recent_patterns:
        pattern_type = pattern['type']

        # Double top - bearish confirmation
        if pattern_type == 'double_top' and ob_type == 'bear':
            # Check if OB formed near the neckline/resistance
            ob_price = (ob['zone_low'] + ob['zone_high']) / 2
            if abs(ob_price - pattern['neckline']) / pattern['neckline'] < 0.02:  # Within 2%
                return {'confirmed': True, 'pattern': pattern, 'strength': 'strong'}

        # Double bottom - bullish confirmation
        elif pattern_type == 'double_bottom' and ob_type == 'bull':
            # Check if OB formed near the neckline/support
            ob_price = (ob['zone_low'] + ob['zone_high']) / 2
            if abs(ob_price - pattern['neckline']) / pattern['neckline'] < 0.02:  # Within 2%
                return {'confirmed': True, 'pattern': pattern, 'strength': 'strong'}

        # Head and shoulders - bearish confirmation
        elif pattern_type == 'head_shoulders' and ob_type == 'bear':
            # Check if OB formed near the neckline
            ob_price = (ob['zone_low'] + ob['zone_high']) / 2
            if abs(ob_price - pattern['neckline']) / pattern['neckline'] < 0.02:  # Within 2%
                return {'confirmed': True, 'pattern': pattern, 'strength': 'strong'}

        # Triangle breakout - direction depends on breakout
        elif 'triangle' in pattern_type:
            ob_price = (ob['zone_low'] + ob['zone_high']) / 2
            breakout_level = pattern['breakout_level']

            # Bullish triangle breakout
            if pattern_type in ['ascending_triangle', 'symmetrical_triangle'] and ob_type == 'bull':
                if abs(ob_price - breakout_level) / breakout_level < 0.03:  # Within 3%
                    return {'confirmed': True, 'pattern': pattern, 'strength': 'strong'}

            # Bearish triangle breakout
            elif pattern_type in ['descending_triangle', 'symmetrical_triangle'] and ob_type == 'bear':
                if abs(ob_price - breakout_level) / breakout_level < 0.03:  # Within 3%
                    return {'confirmed': True, 'pattern': pattern, 'strength': 'strong'}

    return {'confirmed': False, 'pattern': None, 'strength': 'none'}


def run_backtest(df, obs, patterns, symbol):
    balance = START_BALANCE
    equity = [{'timestamp': df.index[0].isoformat(), 'equity': balance}]
    trades = []
    pid = 0

    for ob in obs:
        # Check for pattern confirmation
        pattern_confirm = find_pattern_confirmation(ob, patterns, df)

        # Allow trades with or without pattern confirmation
        # Pattern confirmation gets higher priority/weight
        confirmed_trade = pattern_confirm['confirmed']

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
                trades.append({
                    'id': pid,
                    'symbol': symbol,
                    'type': 'long',
                    'ob_ts': ob['ts'].isoformat(),
                    'entry_ts': df.index[j].isoformat(),
                    'entry': entry,
                    'sl': sl,
                    'tp_partial': tp_partial,
                    'tp_full': tp_full,
                    'exit_ts': result['exit_ts'].isoformat(),
                    'exit': result['exit_price'],
                    'pnl': result['pnl'],
                    'win': result['win'],
                    'pattern_confirmed': 1 if confirmed_trade else 0,
                    'pattern_type': pattern_confirm['pattern']['type'] if confirmed_trade else 'none',
                    'pattern_strength': pattern_confirm['strength'] if confirmed_trade else 'none'
                })
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
                trades.append({
                    'id': pid,
                    'symbol': symbol,
                    'type': 'short',
                    'ob_ts': ob['ts'].isoformat(),
                    'entry_ts': df.index[j].isoformat(),
                    'entry': entry,
                    'sl': sl,
                    'tp_partial': tp_partial,
                    'tp_full': tp_full,
                    'exit_ts': result['exit_ts'].isoformat(),
                    'exit': result['exit_price'],
                    'pnl': result['pnl'],
                    'win': result['win'],
                    'pattern_confirmed': 1 if confirmed_trade else 0,
                    'pattern_type': pattern_confirm['pattern']['type'] if confirmed_trade else 'none',
                    'pattern_strength': pattern_confirm['strength'] if confirmed_trade else 'none'
                })
                pid += 1
                equity.append({'timestamp': result['exit_ts'].isoformat(), 'equity': balance})
                break

    summary = {'starting_balance': START_BALANCE, 'ending_balance': balance, 'trades': len(trades), 'wins': sum(1 for t in trades if t['win']), 'losses': sum(1 for t in trades if not t['win']), 'net_pnl': balance - START_BALANCE}
    return trades, equity, summary


def save_results(trades, equity, summary, symbol):
    # Convert boolean values to integers for JSON serialization
    def convert_booleans(obj):
        if isinstance(obj, dict):
            return {k: convert_booleans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_booleans(item) for item in obj]
        elif isinstance(obj, bool):
            return 1 if obj else 0
        else:
            return obj

    processed_trades = convert_booleans(trades)
    processed_equity = convert_booleans(equity)
    processed_summary = convert_booleans(summary)

    out = {'summary': processed_summary, 'trades': processed_trades, 'equity': processed_equity}
    # Create filename based on symbol
    safe_symbol = symbol.replace('/', '_').replace('=X', '')
    fname = f'smc_patterns_backtest_results_{safe_symbol}.json'
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
    plt.title(f'SMC + Patterns Backtest Equity Curve - {symbol} ({DAYS} days, {TIMEFRAME})')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(alpha=0.3)
    outname = f'smc_patterns_equity_curve_{safe_symbol}.png'
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
        print(f'Fetching {symbol} {TIMEFRAME} data for last {TIMEFRAME} data for last {DAYS} days...')
        df = fetch_forex_ohlcv(symbol, f'{DAYS}d', TIMEFRAME)
    if df.empty:
        print('No data fetched, exiting')
        return

    print(f'Bars fetched: {len(df)}')
    obs = detect_order_blocks(df)
    print(f'Order blocks detected: {len(obs)}')

    patterns = detect_chart_patterns(df)
    print(f'Chart patterns detected: {len(patterns)}')

    # Count pattern types
    pattern_counts = {}
    for p in patterns:
        pattern_counts[p['type']] = pattern_counts.get(p['type'], 0) + 1
    print(f'Pattern breakdown: {pattern_counts}')

    trades, equity, summary = run_backtest(df, obs, patterns, symbol)
    print('=== Summary ===')
    print(f"Starting balance: ${summary['starting_balance']:.2f}")
    print(f"Ending balance:   ${summary['ending_balance']:.2f}")
    print(f"Trades: {summary['trades']}")
    print(f"Wins: {summary['wins']}  Losses: {summary['losses']}")
    if summary['trades'] > 0:
        win_rate = summary['wins'] / summary['trades'] * 100
        print(f"Win Rate: {win_rate:.1f}%")
        avg_win = sum(t['pnl'] for t in trades if t['win']) / summary['wins'] if summary['wins'] > 0 else 0
        avg_loss = sum(t['pnl'] for t in trades if not t['win']) / summary['losses'] if summary['losses'] > 0 else 0
        profit_factor = abs(sum(t['pnl'] for t in trades if t['win']) / sum(t['pnl'] for t in trades if not t['win'])) if summary['losses'] > 0 else float('inf')
        print(f"Avg Win: ${avg_win:.2f}  Avg Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")

        # Pattern confirmation statistics
        confirmed_trades = sum(1 for t in trades if t['pattern_confirmed'])
        confirmed_wins = sum(1 for t in trades if t['pattern_confirmed'] and t['win'])
        unconfirmed_trades = sum(1 for t in trades if not t['pattern_confirmed'])
        unconfirmed_wins = sum(1 for t in trades if not t['pattern_confirmed'] and t['win'])

        print(f"\nPattern Confirmation Stats:")
        print(f"Confirmed trades: {confirmed_trades} ({confirmed_trades/summary['trades']*100:.1f}%)")
        if confirmed_trades > 0:
            print(f"Confirmed win rate: {confirmed_wins/confirmed_trades*100:.1f}%")
        print(f"Unconfirmed trades: {unconfirmed_trades} ({unconfirmed_trades/summary['trades']*100:.1f}%)")
        if unconfirmed_trades > 0:
            print(f"Unconfirmed win rate: {unconfirmed_wins/unconfirmed_trades*100:.1f}%")

    print(f"Net PnL: ${summary['net_pnl']:+.2f}")

    # save_results(trades, equity, summary, symbol)
    try:
        plot_equity(equity, symbol)
    except Exception as e:
        print(f'Could not plot equity curve: {e}')


if __name__ == '__main__':
    main()