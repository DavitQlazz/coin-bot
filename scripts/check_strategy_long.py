#!/usr/bin/env python3
"""
Check single strategy S10 (Balance of Power reversal) for GBPCAD 1h on a long period.

Saves JSON summary to `results/gbpcad_s10_1h_long.json` and an equity curve to
`results/gbpcad_s10_1h_long.png`.
"""
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Parameters
PAIR = 'GBPCAD=X'
INTERVAL = '1h'
# Yahoo / yfinance intraday limit: ~730 days max for 1h/30m/15m. Cap automatically.
REQUESTED_DAYS = 365 * 1  # desired 3 years
if INTERVAL in ('1h', '30m', '15m'):
    DAYS = min(REQUESTED_DAYS, 730)
else:
    DAYS = REQUESTED_DAYS
START_BALANCE = 10000.0
RISK_FRACTION = 0.023  # 2.3% per Pine script
ATR_PERIOD = 14
SL_ATR = 2.8
TP_MULT = 4.0

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


def fetch_forex(symbol, period=f"{DAYS}d", interval=INTERVAL):
    t = yf.Ticker(symbol)
    df = t.history(period=period, interval=interval)
    if df.empty:
        return pd.DataFrame()
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


def atr(df, period=ATR_PERIOD):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def bop(df):
    denom = (df['high'] - df['low']).replace(0, np.nan)
    return (df['close'] - df['open']) / denom


def generate_s10_signals(df):
    s = pd.DataFrame(index=df.index)
    s['close'] = df['close']
    s['bop'] = bop(df)
    # BOP moving average and crossover signal (bop crosses above its MA)
    s['bop_ma'] = s['bop'].rolling(20).mean()
    sig = ((s['bop'].shift(1) <= s['bop_ma'].shift(1)) & (s['bop'] > s['bop_ma'])).astype(int)
    s['atr'] = atr(df)
    return sig, s


def backtest_s10(df, sig, s):
    balance = START_BALANCE
    equity_ts = []
    trades = []
    entry_volumes = []
    idx_list = list(df.index)
    for t in range(len(sig) - 1):
        if sig.iloc[t] == 1:
            entry_pos = t + 1
            if entry_pos >= len(df):
                continue
            entry_idx = idx_list[entry_pos]
            entry_price = df['open'].iloc[entry_pos]
            entry_vol = float(df['volume'].iloc[entry_pos]) if entry_pos < len(df) else 0.0
            entry_volumes.append(entry_vol)
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
                    balance += pnl
                    trades.append({'entry_ts': entry_idx.isoformat(), 'exit_ts': ts.isoformat(), 'pnl': pnl})
                    equity_ts.append((ts.isoformat(), balance))
                    exited = True
                    break
                if high >= tp:
                    pnl = TP_MULT * risk_amount
                    balance += pnl
                    trades.append({'entry_ts': entry_idx.isoformat(), 'exit_ts': ts.isoformat(), 'pnl': pnl})
                    equity_ts.append((ts.isoformat(), balance))
                    exited = True
                    break
            if not exited:
                last_price = df['close'].iloc[-1]
                pnl = (last_price - entry_price) * size
                balance += pnl
                trades.append({'entry_ts': entry_idx.isoformat(), 'exit_ts': df.index[-1].isoformat(), 'pnl': pnl})
                equity_ts.append((df.index[-1].isoformat(), balance))

    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = sum(1 for t in trades if t['pnl'] <= 0)
    gross_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = -sum(t['pnl'] for t in trades if t['pnl'] <= 0)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None
    net = balance - START_BALANCE
    # volume stats
    total_vol = float(sum(entry_volumes)) if entry_volumes else 0.0
    avg_vol = float(total_vol / len(entry_volumes)) if entry_volumes else 0.0
    return {'trades': len(trades), 'wins': wins, 'losses': losses, 'net': net, 'pf': profit_factor, 'total_volume': round(total_vol, 0), 'avg_volume': round(avg_vol, 0), 'trades_list': trades, 'equity': equity_ts}


def plot_equity(equity, outpath):
    if not equity:
        print('No equity points to plot')
        return
    times = [pd.to_datetime(t) for t, _ in equity]
    vals = [v for _, v in equity]
    plt.figure(figsize=(10, 4))
    plt.plot(times, vals, marker='o')
    plt.title(f'{PAIR} S10 equity (1h)')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    print(f'Fetching {PAIR} {INTERVAL} for {DAYS} days...')
    df = fetch_forex(PAIR)
    if df.empty:
        print('No data fetched')
        return
    sig, s = generate_s10_signals(df)
    stats = backtest_s10(df, sig, s)
    out_json = os.path.join(RESULT_DIR, 'gbpcad_s10_1h_long.json')
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=2)
    out_png = os.path.join(RESULT_DIR, 'gbpcad_s10_1h_long.png')
    plot_equity(stats['equity'], out_png)
    print('Done')
    print('Summary:')
    print(f"Trades: {stats['trades']} Wins: {stats['wins']} Losses: {stats['losses']} Net: {stats['net']:.2f} PF: {stats['pf']}")


if __name__ == '__main__':
    main()
