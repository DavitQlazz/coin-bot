#!/usr/bin/env python3
"""
Multi-Strategy Backtester for High Profitability
Tests various trading strategies on forex/crypto data
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
import ta

# Parameters
CRYPTO_EXCHANGE_ID = "binance"
TIMEFRAME = "4h"
DAYS = 180
START_BALANCE = 10000.0
POSITION_FRACTION = 0.02  # fraction of capital risked per trade
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


def calculate_indicators(df):
    """Calculate comprehensive technical indicators"""
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

    # EMA
    df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    # ADX
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()
    df['di_plus'] = adx.adx_pos()
    df['di_minus'] = adx.adx_neg()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    return df


class Strategy:
    """Base strategy class"""
    def __init__(self, name):
        self.name = name

    def generate_signal(self, df, i):
        """Generate trading signal for bar i"""
        return 'HOLD'


class RSIMeanReversion(Strategy):
    """RSI Mean Reversion Strategy"""
    def __init__(self):
        super().__init__("RSI Mean Reversion")

    def generate_signal(self, df, i):
        if i < 14:  # Need warmup
            return 'HOLD'

        rsi = df['rsi'].iloc[i]
        prev_rsi = df['rsi'].iloc[i-1]

        # Buy oversold, sell overbought
        if rsi < 30 and prev_rsi >= 30:
            return 'BUY'
        elif rsi > 70 and prev_rsi <= 70:
            return 'SELL'

        return 'HOLD'


class MACDCrossover(Strategy):
    """MACD Crossover Strategy"""
    def __init__(self):
        super().__init__("MACD Crossover")

    def generate_signal(self, df, i):
        if i < 26:  # MACD warmup
            return 'HOLD'

        macd = df['macd'].iloc[i]
        signal = df['macd_signal'].iloc[i]
        prev_macd = df['macd'].iloc[i-1]
        prev_signal = df['macd_signal'].iloc[i-1]

        # Bullish crossover
        if macd > signal and prev_macd <= prev_signal:
            return 'BUY'
        # Bearish crossover
        elif macd < signal and prev_macd >= prev_signal:
            return 'SELL'

        return 'HOLD'


class BollingerSqueeze(Strategy):
    """Bollinger Band Squeeze Breakout"""
    def __init__(self):
        super().__init__("Bollinger Squeeze")

    def generate_signal(self, df, i):
        if i < 20:  # BB warmup
            return 'HOLD'

        current = df.iloc[i]
        prev = df.iloc[i-1]

        # Look for squeeze (narrowing bands) followed by expansion
        squeeze_threshold = 0.02  # 2% width threshold

        if current['bb_width'] < squeeze_threshold:
            # Squeeze detected, wait for breakout
            if current['close'] > current['bb_high']:
                return 'BUY'
            elif current['close'] < current['bb_low']:
                return 'SELL'

        return 'HOLD'


class ADXTrendFollowing(Strategy):
    """ADX Trend Following Strategy"""
    def __init__(self):
        super().__init__("ADX Trend Following")

    def generate_signal(self, df, i):
        if i < 14:  # ADX warmup
            return 'HOLD'

        adx = df['adx'].iloc[i]
        di_plus = df['di_plus'].iloc[i]
        di_minus = df['di_minus'].iloc[i]
        prev_di_plus = df['di_plus'].iloc[i-1]
        prev_di_minus = df['di_minus'].iloc[i-1]

        # Only trade when trend is strong (ADX > 25)
        if adx > 25:
            # DI crossover signals
            if di_plus > di_minus and prev_di_plus <= prev_di_minus:
                return 'BUY'
            elif di_minus > di_plus and prev_di_minus <= prev_di_plus:
                return 'SELL'

        return 'HOLD'


class EMACrossover(Strategy):
    """EMA Crossover Strategy"""
    def __init__(self):
        super().__init__("EMA Crossover")

    def generate_signal(self, df, i):
        if i < 50:  # EMA warmup
            return 'HOLD'

        ema9 = df['ema_9'].iloc[i]
        ema21 = df['ema_21'].iloc[i]
        prev_ema9 = df['ema_9'].iloc[i-1]
        prev_ema21 = df['ema_21'].iloc[i-1]

        # Golden cross
        if ema9 > ema21 and prev_ema9 <= prev_ema21:
            return 'BUY'
        # Death cross
        elif ema9 < ema21 and prev_ema9 >= prev_ema21:
            return 'SELL'

        return 'HOLD'


class StochasticRSI(Strategy):
    """Stochastic RSI Strategy"""
    def __init__(self):
        super().__init__("Stochastic RSI")

    def generate_signal(self, df, i):
        if i < 14:  # Indicators warmup
            return 'HOLD'

        # Calculate StochRSI manually (simplified)
        rsi = df['rsi'].iloc[i-14:i+1]
        if len(rsi) < 14:
            return 'HOLD'

        stoch_rsi = (rsi - rsi.min()) / (rsi.max() - rsi.min()) * 100
        k = stoch_rsi.iloc[-1]
        d = stoch_rsi.rolling(3).mean().iloc[-1]

        prev_k = (rsi.iloc[-2] - rsi.min()) / (rsi.max() - rsi.min()) * 100
        prev_d = stoch_rsi.rolling(3).mean().iloc[-2]

        # Oversold/overbought levels
        if k < 20 and d < 20 and prev_k >= prev_d:
            return 'BUY'
        elif k > 80 and d > 80 and prev_k <= prev_d:
            return 'SELL'

        return 'HOLD'


def run_strategy_backtest(df, strategy, symbol):
    """Run backtest for a specific strategy"""
    balance = START_BALANCE
    equity = [{'timestamp': df.index[0].isoformat(), 'equity': balance}]
    trades = []
    position = None
    pid = 0

    for i in range(50, len(df)):  # Start after indicator warmup
        current_price = df['close'].iloc[i]
        timestamp = df.index[i]

        # Generate signal
        signal = strategy.generate_signal(df, i)

        # Check for exit conditions if in position
        if position:
            profit_pct = ((current_price / position['entry_price']) - 1) * 100

            # Fixed SL/TP for simplicity (can be improved)
            if position['type'] == 'LONG':
                if profit_pct <= -2.0:  # 2% stop loss
                    pnl = (current_price - position['entry_price']) * position['amount']
                    balance += pnl
                    trades.append({
                        'id': pid, 'symbol': symbol, 'strategy': strategy.name,
                        'type': 'long', 'entry_ts': position['timestamp'].isoformat(),
                        'entry': position['entry_price'], 'exit_ts': timestamp.isoformat(),
                        'exit': current_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
                    })
                    position = None
                    pid += 1
                    equity.append({'timestamp': timestamp.isoformat(), 'equity': balance})
                elif profit_pct >= 4.0:  # 4% take profit
                    pnl = (current_price - position['entry_price']) * position['amount']
                    balance += pnl
                    trades.append({
                        'id': pid, 'symbol': symbol, 'strategy': strategy.name,
                        'type': 'long', 'entry_ts': position['timestamp'].isoformat(),
                        'entry': position['entry_price'], 'exit_ts': timestamp.isoformat(),
                        'exit': current_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
                    })
                    position = None
                    pid += 1
                    equity.append({'timestamp': timestamp.isoformat(), 'equity': balance})
            else:  # SHORT
                if profit_pct >= 2.0:  # 2% stop loss for shorts
                    pnl = (position['entry_price'] - current_price) * position['amount']
                    balance += pnl
                    trades.append({
                        'id': pid, 'symbol': symbol, 'strategy': strategy.name,
                        'type': 'short', 'entry_ts': position['timestamp'].isoformat(),
                        'entry': position['entry_price'], 'exit_ts': timestamp.isoformat(),
                        'exit': current_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
                    })
                    position = None
                    pid += 1
                    equity.append({'timestamp': timestamp.isoformat(), 'equity': balance})
                elif profit_pct <= -4.0:  # 4% take profit for shorts
                    pnl = (position['entry_price'] - current_price) * position['amount']
                    balance += pnl
                    trades.append({
                        'id': pid, 'symbol': symbol, 'strategy': strategy.name,
                        'type': 'short', 'entry_ts': position['timestamp'].isoformat(),
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
            'id': pid, 'symbol': symbol, 'strategy': strategy.name,
            'type': position['type'].lower(), 'entry_ts': position['timestamp'].isoformat(),
            'entry': position['entry_price'], 'exit_ts': df.index[-1].isoformat(),
            'exit': final_price, 'pnl': pnl, 'win': 1 if pnl > 0 else 0
        })
        equity.append({'timestamp': df.index[-1].isoformat(), 'equity': balance})

    summary = {
        'strategy': strategy.name,
        'symbol': symbol,
        'starting_balance': START_BALANCE,
        'ending_balance': balance,
        'trades': len(trades),
        'wins': sum(1 for t in trades if t['win']),
        'losses': sum(1 for t in trades if not t['win']),
        'net_pnl': balance - START_BALANCE,
        'win_rate': (sum(1 for t in trades if t['win']) / len(trades) * 100) if trades else 0
    }

    return trades, equity, summary


def main():
    symbol = 'EURUSD=X'  # Default forex pair
    data_source = 'forex'

    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        if symbol.endswith('=X'):
            data_source = 'forex'
        elif '/' in symbol:
            data_source = 'crypto'
        else:
            data_source = 'forex'
            symbol = symbol + '=X'

    print(f"Testing multiple strategies on {symbol}")
    print(f"Data source: {data_source.upper()}")
    print(f"Timeframe: {TIMEFRAME}, Period: {DAYS} days")

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
        print('No data fetched, exiting')
        return

    print(f"Bars fetched: {len(df)}")

    # Calculate indicators
    df = calculate_indicators(df)

    # Define strategies to test
    strategies = [
        RSIMeanReversion(),
        MACDCrossover(),
        BollingerSqueeze(),
        ADXTrendFollowing(),
        EMACrossover(),
        StochasticRSI()
    ]

    results = []

    # Run backtest for each strategy
    for strategy in strategies:
        print(f"\nTesting {strategy.name}...")
        trades, equity, summary = run_strategy_backtest(df, strategy, symbol)
        results.append(summary)

        print(f"  Trades: {summary['trades']}")
        print(f"  Wins: {summary['wins']}  Losses: {summary['losses']}")
        print(f"  Win Rate: {summary['win_rate']:.1f}%")
        print(f"  Net P/L: ${summary['net_pnl']:+.2f}")
        print(f"  End Balance: ${summary['ending_balance']:.2f}")

    # Print comparison
    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON - {symbol} ({TIMEFRAME}, {DAYS} days)")
    print('='*80)
    print(f"{'Strategy':<20} {'Trades':<6} {'Win%':<6} {'Net P/L':<10} {'End Balance':<12}")
    print('-'*80)

    for result in results:
        print(f"{result['strategy']:<20} {result['trades']:<6} {result['win_rate']:<6.1f} "
              f"${result['net_pnl']:<9.0f} ${result['ending_balance']:<11.0f}")

    # Find best performer
    if results:
        best = max(results, key=lambda x: x['net_pnl'])
        print('-'*80)
        print(f"Best Strategy: {best['strategy']} (${best['net_pnl']:+.0f} P/L, {best['win_rate']:.1f}% win rate)")

        # Save best strategy results
        safe_symbol = symbol.replace('/', '_').replace('=X', '')
        safe_strategy = best['strategy'].replace(' ', '_').lower()
        fname = f'best_strategy_{safe_strategy}_{safe_symbol}.json'

        # Get trades and equity for best strategy
        best_strategy = next(s for s in strategies if s.name == best['strategy'])
        trades, equity, _ = run_strategy_backtest(df, best_strategy, symbol)

        with open(fname, 'w') as f:
            json.dump({
                'summary': best,
                'trades': trades,
                'equity': equity
            }, f, indent=2)

        print(f"Best strategy results saved to {fname}")


if __name__ == '__main__':
    main()