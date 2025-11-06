#!/usr/bin/env python3
"""
Momentum strategies comparison (10 variants)

This script implements simplified versions of 10 momentum strategies,
backtests them on a few forex pairs and produces a comparison summary.

Notes:
- Signals are simplified approximations to keep the tests fast and deterministic.
- Trade sizing: fixed risk fraction per trade with ATR-based SL and 2:1 TP.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import math
import json
from datetime import datetime
import os

# Config
PAIRS = ["GBPCAD=X", "USDCAD=X", "EURJPY=X"]
TIMEFRAMES = ["1h", "4h"]
DAYS = 365
START_BALANCE = 10000.0
RISK_FRACTION = 0.012  # 1.2% per trade
ATR_PERIOD = 14
SL_ATR = 1.5
TP_MULT = 2.0

# Load strategy configuration (names, meta)
STRATEGIES = {}
cfg_path = os.path.join(os.path.dirname(__file__), 'strategies_config.json')
if os.path.exists(cfg_path):
    try:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
            STRATEGIES = {str(item['id']): item for item in cfg}
    except Exception:
        STRATEGIES = {}
else:
    STRATEGIES = {}


def fetch_forex(symbol, period=f"{DAYS}d", interval="1h"):
    t = yf.Ticker(symbol)
    try:
        df = t.history(period=period, interval=interval)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.reset_index()
    df.columns = df.columns.str.lower()
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for c in required:
        if c not in df.columns:
            df[c] = 0
    df = df[required]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(period).mean()


def atr(df, period=ATR_PERIOD):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def macd(df):
    macd_line = ema(df['close'], 12) - ema(df['close'], 26)
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def stoch_rsi(df, rsi_period=14, stoch_period=14):
    r = rsi(df['close'], rsi_period)
    min_r = r.rolling(stoch_period).min()
    max_r = r.rolling(stoch_period).max()
    stochr = (r - min_r) / (max_r - min_r)
    return stochr


def cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.fabs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)


def williams_r(df, period=14):
    high = df['high'].rolling(period).max()
    low = df['low'].rolling(period).min()
    wr = (high - df['close']) / (high - low) * -100
    return wr


def tsi(df, long=25, short=13, signal=7):
    m = df['close'].diff()
    em1 = m.ewm(span=short, adjust=False).mean()
    em2 = em1.ewm(span=long, adjust=False).mean()
    abs_m = m.abs()
    ema1 = abs_m.ewm(span=short, adjust=False).mean()
    ema2 = ema1.ewm(span=long, adjust=False).mean()
    tsi_series = 100 * (em2 / ema2)
    sig = tsi_series.ewm(span=signal, adjust=False).mean()
    return tsi_series, sig


def roc(df, period=12):
    return df['close'].pct_change(periods=period) * 100


def ultimate_osc(df):
    avg7 = (df['close'] - df['low']).rolling(7).sum()
    avg14 = (df['close'] - df['low']).rolling(14).sum()
    avg28 = (df['close'] - df['low']).rolling(28).sum()
    uo = (4 * avg7 + 2 * avg14 + avg28) / (4 + 2 + 1)
    return uo


def awesome_osc(df):
    return df['close'].rolling(5).mean() - df['close'].rolling(34).mean()


def bop(df):
    denom = (df['high'] - df['low']).replace(0, np.nan)
    return (df['close'] - df['open']) / denom


def generate_signals(df, strategy_id):
    s = pd.DataFrame(index=df.index)
    s['close'] = df['close']
    # indicators
    s['atr'] = atr(df)
    s['rsi'] = rsi(df['close'])
    macd_line, macd_sig, macd_hist = macd(df)
    s['macd_hist'] = macd_hist
    s['stochrsi'] = stoch_rsi(df)
    s['cci'] = cci(df)
    s['wr'] = williams_r(df)
    tsi_s, tsi_sig = tsi(df)
    s['tsi'] = tsi_s
    s['roc'] = roc(df)
    s['uo'] = ultimate_osc(df)
    s['ao'] = awesome_osc(df)
    s['bop'] = bop(df)

    sig = pd.Series(0, index=s.index)

    if strategy_id == 1:
        # RSI Divergence Momentum approximation: RSI crosses above 30 and MACD hist > 0
        sig = ((s['rsi'].shift(1) < 30) & (s['rsi'] > 30) & (s['macd_hist'] > 0)).astype(int)
    elif strategy_id == 2:
        # Stochastic RSI Breakout: stochrsi crosses above 0.2
        sig = ((s['stochrsi'].shift(1) < 0.2) & (s['stochrsi'] > 0.2)).astype(int)
    elif strategy_id == 3:
        # MACD Trend Following: MACD hist > 0 and close > ema50
        s['ema50'] = ema(s['close'], 50)
        sig = ((s['macd_hist'] > 0) & (s['close'] > s['ema50'])).astype(int)
    elif strategy_id == 4:
        # CCI Extreme Reversal: CCI < -200 then crosses up
        sig = ((s['cci'].shift(1) < -200) & (s['cci'] > -200)).astype(int)
    elif strategy_id == 5:
        # Williams %R Bounce: WR < -80 then cross up
        sig = ((s['wr'].shift(1) < -80) & (s['wr'] > -80)).astype(int)
    elif strategy_id == 6:
        # TSI signal line cross: tsi crosses above its signal (approx)
        tsi_sig = tsi_s.ewm(span=7, adjust=False).mean()
        sig = ((s['tsi'].shift(1) < tsi_sig.shift(1)) & (s['tsi'] > tsi_sig)).astype(int)
    elif strategy_id == 7:
        # ROC Acceleration: ROC > 5 and price > ema20
        s['ema20'] = ema(s['close'], 20)
        sig = ((s['roc'] > 5) & (s['close'] > s['ema20'])).astype(int)
    elif strategy_id == 8:
        # Ultimate Oscillator: uo in low quantile and rising
        low_q = s['uo'].quantile(0.2)
        sig = ((s['uo'].shift(1) < s['uo']) & (s['uo'] < low_q)).astype(int)
    elif strategy_id == 9:
        # Awesome Oscillator cross above zero
        sig = ((s['ao'].shift(1) < 0) & (s['ao'] > 0)).astype(int)
    elif strategy_id == 10:
        # Balance of Power extreme negative then cross up
        sig = ((s['bop'].shift(1) < -0.4) & (s['bop'] > -0.4)).astype(int)

    return sig, s


def backtest_signals(df, sig, s):
    balance = START_BALANCE
    trades = []
    idx_list = list(df.index)
    for t in range(len(sig) - 1):
        if sig.iloc[t] == 1:
            # entry at next bar open
            entry_pos = t + 1
            if entry_pos >= len(df):
                continue
            entry_idx = idx_list[entry_pos]
            entry_price = df['open'].iloc[entry_pos]
            atr_val = s['atr'].iloc[entry_pos]
            if np.isnan(atr_val) or atr_val == 0:
                continue
            sl = entry_price - SL_ATR * atr_val
            tp = entry_price + TP_MULT * (entry_price - sl)
            risk_per_unit = entry_price - sl
            risk_amount = balance * RISK_FRACTION
            size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

            exited = False
            for f in range(entry_pos + 1, len(df)):
                low = df['low'].iloc[f]
                high = df['high'].iloc[f]
                ts = idx_list[f]
                if low <= sl:
                    pnl = -risk_amount
                    trades.append({'entry_ts': entry_idx.isoformat(), 'exit_ts': ts.isoformat(), 'pnl': pnl})
                    balance += pnl
                    exited = True
                    break
                if high >= tp:
                    pnl = TP_MULT * risk_amount
                    trades.append({'entry_ts': entry_idx.isoformat(), 'exit_ts': ts.isoformat(), 'pnl': pnl})
                    balance += pnl
                    exited = True
                    break
            if not exited:
                last_price = df['close'].iloc[-1]
                pnl = (last_price - entry_price) * size
                trades.append({'entry_ts': entry_idx.isoformat(), 'exit_ts': df.index[-1].isoformat(), 'pnl': pnl})
                balance += pnl

    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = sum(1 for t in trades if t['pnl'] <= 0)
    gross_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = -sum(t['pnl'] for t in trades if t['pnl'] <= 0)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float('inf')
    net = balance - START_BALANCE
    return {'trades': len(trades), 'wins': wins, 'losses': losses, 'net': net, 'pf': profit_factor}


def run_all():
    results = {}
    for pair in PAIRS:
        results[pair] = {}
        for tf in TIMEFRAMES:
            print(f"Running pair {pair} timeframe {tf}...")
            df = fetch_forex(pair, period=f"{DAYS}d", interval=tf)
            if df.empty:
                print(f"No data for {pair} {tf}")
                continue
            results[pair][tf] = {}
            for sid in range(1, 11):
                sig, s = generate_signals(df, sid)
                stats = backtest_signals(df, sig, s)
                key = f"S{sid}"
                results[pair][tf][key] = stats
                name = STRATEGIES.get(str(sid), {}).get('name', key)
                print(f"  {name} ({key}): trades={stats['trades']} wins={stats['wins']} net={stats['net']:.2f} pf={stats['pf']:.2f}")

    with open('momentum_strategies_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nCOMPARISON SUMMARY:')
    rows = []
    for pair in results:
        for tf in results[pair]:
            for sid, stats in results[pair][tf].items():
                name = STRATEGIES.get(sid.replace('S', ''), {}).get('name', sid)
                rows.append((pair, tf, sid, name, stats['trades'], stats['wins'], stats['net'], stats['pf']))
    dfc = pd.DataFrame(rows, columns=['pair', 'tf', 'strategy_key', 'strategy_name', 'trades', 'wins', 'net', 'pf'])
    print(dfc.sort_values(['pf', 'net'], ascending=[False, False]).to_string(index=False))


if __name__ == '__main__':
    run_all()
