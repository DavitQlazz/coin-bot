#!/usr/bin/env python3
"""
Run S10 backtest for a single symbol and save results.
"""
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Parameters
SYMBOL = 'GBPJPY=X'
INTERVAL = '4h'
DAYS = 360
START_BALANCE = 1000.0
RISK_FRACTION = 0.001
ATR_PERIOD = 14
SL_ATR = 0.5
TP_MULT = 2.0
USE_TRAILING_STOP = True

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULT_DIR = os.path.join(ROOT, 'results')
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


def bop(df):
    denom = (df['high'] - df['low']).replace(0, np.nan)
    return (df['close'] - df['open']) / denom


def atr(df, period=ATR_PERIOD):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_vwap(df, window=20):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df['volume'].fillna(0.0)
    if vol.sum() == 0:
        return tp.rolling(window).mean()
    parts = []
    for date, g in df.groupby(df.index.date):
        g_tp = ((g['high'] + g['low'] + g['close']) / 3.0)
        g_vol = g['volume']
        ctv = (g_tp * g_vol).cumsum()
        cv = g_vol.cumsum()
        v = ctv / cv
        v.index = g.index
        parts.append(v)
    return pd.concat(parts).reindex(df.index).fillna(method='ffill')

def generate_s10_signals(df):
    s = pd.DataFrame(index=df.index)
    s['close'] = df['close']
    s['bop'] = bop(df)
    s['bop_ma'] = s['bop'].rolling(20).mean()
    s['vwap'] = compute_vwap(df)
    sig_long = ((s['bop'].shift(1) <= s['bop_ma'].shift(1)) & (s['bop'] > s['bop_ma']) & (df['close'] > s['vwap'])).astype(int)
    sig_short = ((s['bop'].shift(1) >= s['bop_ma'].shift(1)) & (s['bop'] < s['bop_ma']) & (df['close'] < s['vwap'])).astype(int)
    s['atr'] = atr(df)
    return sig_long, sig_short, s

# --- New: Confluence-based strategy ---
def generate_confluence_signals(df):
    s = pd.DataFrame(index=df.index)
    s['close'] = df['close']
    s['bop'] = bop(df)
    s['bop_ma'] = s['bop'].rolling(20).mean()
    s['vwap'] = compute_vwap(df)
    s['atr'] = atr(df)
    s['sma_20'] = df['close'].rolling(20).mean()
    s['sma_50'] = df['close'].rolling(50).mean()
    s['adx'] = pd.Series(np.nan, index=df.index)
    # Simple ADX calculation (not optimized)
    def calc_adx(df, n=14):
        up = df['high'].diff()
        down = -df['low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr_ = tr.rolling(n).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(n).sum() / atr_
        minus_di = 100 * pd.Series(minus_dm).rolling(n).sum() / atr_
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(n).mean()
        return adx
    s['adx'] = calc_adx(df)
    # Entry rules:
    # Long: bop cross up, price > vwap, price > sma20 > sma50, adx between 15 and 35, atr above median
    # Short: bop cross down, price < vwap, price < sma20 < sma50, adx between 15 and 35, atr above median
    # Remove ATR and ADX filters, allow any trend direction
    sig_long = (
        (s['bop'].shift(1) <= s['bop_ma'].shift(1)) &
        (s['bop'] > s['bop_ma']) &
        (s['close'] > s['vwap'])
    ).astype(int)
    sig_short = (
        (s['bop'].shift(1) >= s['bop_ma'].shift(1)) &
        (s['bop'] < s['bop_ma']) &
        (s['close'] < s['vwap'])
    ).astype(int)
    return sig_long, sig_short, s


def backtest_s10(df, sig_long, sig_short, s):
    balance = START_BALANCE
    equity_ts = []
    trades = []
    idx_list = list(df.index)
    n = len(df)
    entry_volumes = []  # track volumes at entries

    for t in range(n - 1):
        # LONG
        if sig_long.iloc[t] == 1:
            entry_pos = t + 1
            if entry_pos >= n:
                continue
            entry_idx = idx_list[entry_pos]
            entry_price = df['open'].iloc[entry_pos]
            entry_vol = df['volume'].iloc[entry_pos]
            entry_volumes.append(entry_vol)
            atr_val = s['atr'].iloc[entry_pos]
            if pd.isna(atr_val) or atr_val == 0:
                continue
            sl = entry_price - SL_ATR * atr_val
            tp = entry_price + TP_MULT * (entry_price - sl)
            risk_per_unit = entry_price - sl
            risk_amount = balance * RISK_FRACTION
            size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

            exited = False
            max_price = entry_price
            for f in range(entry_pos + 1, n):
                low = df['low'].iloc[f]
                high = df['high'].iloc[f]
                ts = idx_list[f]
                if USE_TRAILING_STOP:
                    # Move stop up if new high
                    if high > max_price:
                        max_price = high
                    new_sl = max(sl, max_price - SL_ATR * atr_val)
                    sl = new_sl
                if low <= sl:
                    pnl = (sl - entry_price) * size
                    balance += pnl
                    trades.append({'side': 'long', 'entry_ts': entry_idx.isoformat(), 'exit_ts': ts.isoformat(), 'pnl': pnl, 'size': size, 'entry_price': entry_price, 'exit_price': sl})
                    equity_ts.append((ts.isoformat(), balance))
                    exited = True
                    break
                if not USE_TRAILING_STOP and high >= tp:
                    pnl = (tp - entry_price) * size
                    balance += pnl
                    trades.append({'side': 'long', 'entry_ts': entry_idx.isoformat(), 'exit_ts': ts.isoformat(), 'pnl': pnl, 'size': size, 'entry_price': entry_price, 'exit_price': tp})
                    equity_ts.append((ts.isoformat(), balance))
                    exited = True
                    break
            if not exited:
                last_price = df['close'].iloc[-1]
                pnl = (last_price - entry_price) * size
                balance += pnl
                trades.append({'side': 'long', 'entry_ts': entry_idx.isoformat(), 'exit_ts': df.index[-1].isoformat(), 'pnl': pnl, 'size': size, 'entry_price': entry_price, 'exit_price': last_price})
                equity_ts.append((df.index[-1].isoformat(), balance))

        # SHORT
        if sig_short.iloc[t] == 1:
            entry_pos = t + 1
            if entry_pos >= n:
                continue
            entry_idx = idx_list[entry_pos]
            entry_price = df['open'].iloc[entry_pos]
            entry_vol = df['volume'].iloc[entry_pos]
            entry_volumes.append(entry_vol)
            atr_val = s['atr'].iloc[entry_pos]
            if pd.isna(atr_val) or atr_val == 0:
                continue
            sl = entry_price + SL_ATR * atr_val  # stop above entry
            tp = entry_price - TP_MULT * (sl - entry_price)
            risk_per_unit = sl - entry_price
            risk_amount = balance * RISK_FRACTION
            size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

            exited = False
            min_price = entry_price
            for f in range(entry_pos + 1, n):
                low = df['low'].iloc[f]
                high = df['high'].iloc[f]
                ts = idx_list[f]
                if USE_TRAILING_STOP:
                    # Move stop down if new low
                    if low < min_price:
                        min_price = low
                    new_sl = min(sl, min_price + SL_ATR * atr_val)
                    sl = new_sl
                if high >= sl:
                    pnl = (entry_price - sl) * size
                    balance += pnl
                    trades.append({'side': 'short', 'entry_ts': entry_idx.isoformat(), 'exit_ts': ts.isoformat(), 'pnl': pnl, 'size': size, 'entry_price': entry_price, 'exit_price': sl})
                    equity_ts.append((ts.isoformat(), balance))
                    exited = True
                    break
                if not USE_TRAILING_STOP and low <= tp:
                    pnl = (entry_price - tp) * size
                    balance += pnl
                    trades.append({'side': 'short', 'entry_ts': entry_idx.isoformat(), 'exit_ts': ts.isoformat(), 'pnl': pnl, 'size': size, 'entry_price': entry_price, 'exit_price': tp})
                    equity_ts.append((ts.isoformat(), balance))
                    exited = True
                    break
            if not exited:
                last_price = df['close'].iloc[-1]
                pnl = (entry_price - last_price) * size
                balance += pnl
                trades.append({'side': 'short', 'entry_ts': entry_idx.isoformat(), 'exit_ts': df.index[-1].isoformat(), 'pnl': pnl, 'size': size, 'entry_price': entry_price, 'exit_price': last_price})
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
    return {'symbol': SYMBOL, 'interval': INTERVAL, 'days': DAYS, 'trades': len(trades), 'wins': wins, 'losses': losses, 'net': round(net,2), 'pf': profit_factor, 'total_volume': round(total_vol,0), 'avg_volume': round(avg_vol,0), 'trades_list': trades, 'equity': equity_ts}


def plot_equity(equity, outpath):
    if not equity:
        print('No equity points to plot')
        return
    times = [pd.to_datetime(t) for t, _ in equity]
    vals = [v for _, v in equity]
    plt.figure(figsize=(10, 4))
    plt.plot(times, vals, marker='o')
    plt.title(f'{SYMBOL} S10 equity ({INTERVAL})')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    import sys
    symbol = SYMBOL
    interval = INTERVAL
    days = DAYS
    strategy = 's10'  # default
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    if len(sys.argv) > 2:
        interval = sys.argv[2]
    if len(sys.argv) > 3:
        days = int(sys.argv[3])
    if len(sys.argv) > 4:
        strategy = sys.argv[4].lower()
    print(f'Fetching {symbol} {interval} for {days} days...')
    df = fetch_forex(symbol, period=f"{days}d", interval=interval)
    if df.empty:
        print('No data fetched')
        return
    if strategy == 'confluence':
        sig_long, sig_short, s = generate_confluence_signals(df)
        out_prefix = f"{symbol.replace('=','').lower()}_confluence_{interval}_{days}d"
    else:
        sig_long, sig_short, s = generate_s10_signals(df)
        out_prefix = f"{symbol.replace('=','').lower()}_s10_{interval}_{days}d"
    stats = backtest_s10(df, sig_long, sig_short, s)
    out_json = os.path.join(RESULT_DIR, f"{out_prefix}.json")
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=2)
    out_png = os.path.join(RESULT_DIR, f"{out_prefix}.png")
    plot_equity(stats['equity'], out_png)
    # Calculate pips for each trade and total
    def pip_factor(symbol):
        # Most forex pairs: 0.0001, JPY pairs: 0.01
        if symbol.endswith('JPY=X') or symbol.endswith('JPY'): return 0.01
        return 0.0001
    pfac = pip_factor(symbol.upper())
    total_pips = 0.0
    trades_with_pips = []
    for t in stats['trades_list']:
        # Calculate pips directly from entry and exit price
        entry_price = t.get('entry_price', 0)
        exit_price = t.get('exit_price', 0)
        if entry_price and exit_price:
            if t['side'] == 'long':
                pips = (exit_price - entry_price) / pfac
            else:  # short
                pips = (entry_price - exit_price) / pfac
        else:
            pips = 0.0
        total_pips += pips
        trades_with_pips.append({**t, 'pips': pips})
    print('Done')
    print('Summary:')
    print(f"Trades: {stats['trades']} Wins: {stats['wins']} Losses: {stats['losses']} Net: {stats['net']:.2f} PF: {stats['pf']} Total Pips: {total_pips:.1f}")
    # Output all trades to CSV file in results directory
    if stats['trades_list']:
        import csv
        trades_csv = os.path.join(RESULT_DIR, f"{out_prefix}_trades.csv")
        with open(trades_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Side', 'Entry', 'Exit', 'PnL', 'Pips'])
            for t in trades_with_pips:
                writer.writerow([t['side'], t['entry_ts'], t['exit_ts'], f"{t['pnl']:.2f}", f"{t['pips']:.1f}"])
        print(f"All trades written to {trades_csv}")


if __name__ == '__main__':
    main()
