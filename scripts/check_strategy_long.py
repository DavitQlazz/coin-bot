#!/usr/bin/env python3
"""
Run S10 (Balance of Power reversal) for GBPCAD=X 1h for a long period (capped at 730 days for Yahoo intraday limits).
Saves results to results/gbpcad_s10_1h_long.json and a PNG equity plot.
"""
import os, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

PAIR = 'GBPCAD=X'
TF = '1h'
DAYS = 1095

def fetch(pair, tf, days):
    # Yahoo intraday limit: cap to 730 days for 1h/30m/15m
    if tf in ['1h','30m','15m'] and days>730:
        print('Capping days to 730 for intraday', tf)
        days = 730
    period = f"{days}d"
    df = yf.download(pair, period=period, interval=tf, progress=False)
    if df.empty:
        print('No data for', pair, tf, period)
        return None
    df = df.dropna()
    return df

def bop(df):
    # Balance of Power = (close - open) / (high - low)
    bor = (df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0,np.nan)
    return bor.fillna(0)

def generate_signals(df):
    df = df.copy()
    df['bop'] = bop(df)
    df['bop_ma'] = df['bop'].rolling(20).mean()
    # signal: bop crosses above its MA -> long entry signal
    df['sig'] = ((df['bop'] > df['bop_ma']) & (df['bop'].shift(1) <= df['bop_ma'].shift(1))).astype(int)
    return df

def backtest(df):
    START = 10000.0
    BAL = START
    RISK = 0.012
    ATR_PERIOD = 14
    SL_ATR = 1.5
    TP_MULT = 2.0

    df = df.copy()
    df['atr'] = df['High'].combine(df['Low'], max) - df['Low']
    df['atr'] = df['atr'].rolling(ATR_PERIOD).mean().fillna(method='bfill')

    trades = []
    equity = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        if df['sig'].iloc[i-1]==1:
            entry = row['Open']
            sl = entry - SL_ATR * row['atr']
            tp = entry + TP_MULT * (entry - sl)
            risk_amount = BAL * RISK
            qty = risk_amount / (entry - sl) if (entry - sl)>0 else 0
            # simulate next bars until exit
            exit_price = None
            exit_idx = None
            for j in range(i, len(df)):
                high = df['High'].iloc[j]
                low = df['Low'].iloc[j]
                if low <= sl:
                    exit_price = sl
                    exit_idx = j
                    break
                if high >= tp:
                    exit_price = tp
                    exit_idx = j
                    break
            if exit_price is None:
                # close at last available close
                exit_price = df['Close'].iloc[-1]
                exit_idx = len(df)-1
            pnl = (exit_price - entry) * qty
            BAL += pnl
            trades.append({'entry_time': df.index[i].isoformat(), 'exit_time': df.index[exit_idx].isoformat(), 'entry': entry, 'exit': exit_price, 'pnl': pnl})
            equity.append((df.index[exit_idx].isoformat(), BAL))

    wins = sum(1 for t in trades if t['pnl']>0)
    net = sum(t['pnl'] for t in trades)
    pf = sum(t['pnl'] for t in trades if t['pnl']>0) / (-sum(t['pnl'] for t in trades if t['pnl']<0)) if any(t['pnl']<0 for t in trades) else None

    out = {'pair': PAIR, 'tf': TF, 'trades': len(trades), 'wins': wins, 'net': round(net,2), 'pf': round(pf,12) if pf else None, 'equity': equity, 'trades_list': trades}
    return out

def plot_equity(eq_fp):
    with open(eq_fp) as f:
        d = json.load(f)
    eq = d.get('equity', [])
    if not eq:
        return
    dates = [pd.to_datetime(t) for t,_ in eq]
    vals = [v for _,v in eq]
    plt.figure(figsize=(10,4))
    plt.plot(dates, vals)
    plt.title(f"{PAIR} {TF} S10 equity")
    plt.savefig(os.path.join(RESULTS,'gbpcad_s10_1h_long.png'))
    plt.close()

def main():
    df = fetch(PAIR, TF, DAYS)
    if df is None:
        return
    df2 = generate_signals(df)
    res = backtest(df2)
    out_fp = os.path.join(RESULTS,'gbpcad_s10_1h_long.json')
    with open(out_fp,'w') as f:
        json.dump(res,f,indent=2)
    plot_equity(out_fp)
    print('Done. Saved', out_fp)

if __name__=='__main__':
    main()
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
REQUESTED_DAYS = 365 * 3  # desired 3 years
if INTERVAL in ('1h', '30m', '15m'):
    DAYS = min(REQUESTED_DAYS, 730)
else:
    DAYS = REQUESTED_DAYS
START_BALANCE = 10000.0
RISK_FRACTION = 0.012
ATR_PERIOD = 14
SL_ATR = 1.5
TP_MULT = 2.0

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
    sig = ((s['bop'].shift(1) < -0.4) & (s['bop'] > -0.4)).astype(int)
    s['atr'] = atr(df)
    return sig, s


def backtest_s10(df, sig, s):
    balance = START_BALANCE
    equity_ts = []
    trades = []
    idx_list = list(df.index)
    for t in range(len(sig) - 1):
        if sig.iloc[t] == 1:
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
    return {'trades': len(trades), 'wins': wins, 'losses': losses, 'net': net, 'pf': profit_factor, 'trades_list': trades, 'equity': equity_ts}


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
