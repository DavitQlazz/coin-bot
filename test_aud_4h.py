#!/usr/bin/env python3
"""
Test High-WR Scalping Strategy on 4H Timeframe
Evaluates top 4 AUD pairs (AUDNZD, AUDCAD, AUDHKD, AUDEUR) on 4h candles
with 30-day backtest period and extended hold time (240m = 4 hours).
"""

import os
import sys
import json
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings('ignore')

class HighWRScalpingBot4H:
    def __init__(self, pair, atr_sl_mult=1.4, atr_tp_mult=2.4, max_hold_minutes=240):
        self.pair = pair
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_hold_time = max_hold_minutes
        self.starting_capital = 1000.0
        self.capital = self.starting_capital
        self.trades = []
        print(f"✅ High Win-Rate Scalping Bot initialized")
        print(f"   Pair: {pair}")
        print(f"   ATR SL: {atr_sl_mult}x ATR")
        print(f"   ATR TP: {atr_tp_mult}x ATR")
        print(f"   Risk-Reward: 1:{atr_tp_mult/atr_sl_mult:.1f}")
        print(f"   Max Hold Time: {max_hold_minutes} minutes")
        print(f"   Target Win Rate: 55%\n")

    def fetch_data(self, period='30d', interval='4h'):
        """Fetch 4h data from Yahoo Finance."""
        try:
            print(f"📊 Fetching {interval} data for {self.pair}...")
            df = yf.download(self.pair, period=period, interval=interval, progress=False)
            if df.empty or len(df) < 50:
                print(f"❌ Error: No data returned for {self.pair}")
                return None
            # Reset index to ensure proper datetime index
            df = df.reset_index()
            df = df.set_index('Datetime')
            print(f"✅ Fetched {len(df)} candles")
            return df
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None

    def calculate_indicators(self, df):
        """Calculate all indicators for entry/exit logic."""
        df = df.copy()
        
        # ATR (volatility) - manual calculation
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Bollinger Bands (mean reversion)
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['BB_Mid'] + (bb_std * 2)
        df['BB_Low'] = df['BB_Mid'] - (bb_std * 2)
        
        # RSI (momentum) - manual calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Simple Stochastic (momentum)
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # EMA (trend)
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # MACD (manual calculation)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # ADX (trend strength) - simplified
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = df['TR']
        plus_di = 100 * plus_dm.rolling(window=14).mean() / df['ATR']
        minus_di = 100 * minus_dm.rolling(window=14).mean() / df['ATR']
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(window=14).mean()
        
        # Volume ratio
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-10)
        
        # VWAP (20-period rolling)
        df['VWAP'] = self._calculate_vwap(df, window=20)
        
        # ATR percentile
        df['ATR_Percentile'] = df['ATR'].rolling(window=20).apply(
            lambda x: (x.iloc[-1] > x.quantile(0.1)) and (x.iloc[-1] < x.quantile(0.9))
        )
        
        return df.dropna()

    def _calculate_vwap(self, df, window=20):
        """Calculate rolling VWAP."""
        try:
            from ta.volume import VolumeWeightedAveragePrice
            vwap = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'], window=window)
            return vwap.volume_weighted_average_price()
        except:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            cum_vol_tp = (typical_price * df['Volume']).rolling(window=window).sum()
            cum_vol = df['Volume'].rolling(window=window).sum()
            return cum_vol_tp / cum_vol

    def generate_signal(self, df, index, relaxed=False):
        """Generate entry signal with graded scoring."""
        if index < 50:
            return 'hold', None, None
        
        row = df.iloc[index]
        prev_row = df.iloc[index - 1]
        
        # Skip invalid data
        if pd.isna(row['ATR']) or pd.isna(row['RSI']) or row['ATR'] == 0:
            return 'hold', None, None
        
        # Calculate SL and TP
        atr = row['ATR']
        sl_pct = self.atr_sl_mult * atr / row['Close']
        tp_pct = self.atr_tp_mult * atr / row['Close']
        
        # ===== LONG SCORING =====
        long_score = 0
        
        # 1. BB extreme (2x weight)
        if row['Close'] < row['BB_Low'] * 1.01:
            long_score += 2
        
        # 2. RSI oversold
        if row['RSI'] < 30:
            long_score += 1
        
        # 3. Stochastic oversold
        if row['Stoch_K'] < 20:
            long_score += 1
        
        # 4. EMA bias (price above shorter EMAs)
        if row['Close'] > row['EMA_12']:
            long_score += 1
        
        # 5. Mean reversion (price < EMA50)
        if row['Close'] < row['EMA_50']:
            long_score += 1
        
        # 6. MACD positive slope
        if row['MACD'] > row['MACD_Signal']:
            long_score += 1
        
        # 7. ADX in range (15–30)
        if 15 <= row['ADX'] <= 30:
            long_score += 1
        
        # 8. Volume above average
        if row['Volume_Ratio'] > 1.0:
            long_score += 1
        
        # 9. VWAP confluence
        if row['Close'] > row['VWAP'] - (atr * 0.0001):
            long_score += 1
        
        # ===== SHORT SCORING =====
        short_score = 0
        
        # 1. BB extreme (2x weight)
        if row['Close'] > row['BB_High'] * 0.99:
            short_score += 2
        
        # 2. RSI overbought
        if row['RSI'] > 70:
            short_score += 1
        
        # 3. Stochastic overbought
        if row['Stoch_K'] > 80:
            short_score += 1
        
        # 4. EMA bias (price below shorter EMAs)
        if row['Close'] < row['EMA_12']:
            short_score += 1
        
        # 5. Mean reversion (price > EMA50)
        if row['Close'] > row['EMA_50']:
            short_score += 1
        
        # 6. MACD negative slope
        if row['MACD'] < row['MACD_Signal']:
            short_score += 1
        
        # 7. ADX in range (15–30)
        if 15 <= row['ADX'] <= 30:
            short_score += 1
        
        # 8. Volume above average
        if row['Volume_Ratio'] > 1.0:
            short_score += 1
        
        # 9. VWAP confluence
        if row['Close'] < row['VWAP'] + (atr * 0.0001):
            short_score += 1
        
        # ===== DECISION =====
        required_score = 4 if relaxed else 5
        
        if long_score >= required_score and long_score > short_score:
            return 'buy', sl_pct, tp_pct
        elif short_score >= required_score and short_score > long_score:
            return 'sell', sl_pct, tp_pct
        else:
            return 'hold', None, None

    def backtest(self, period='30d', interval='4h'):
        """Run backtest on 4h data."""
        df = self.fetch_data(period=period, interval=interval)
        if df is None:
            return None
        
        df = self.calculate_indicators(df)
        
        print(f"\n{'='*70}")
        print(f"HIGH WIN-RATE SCALPING BACKTEST: {self.pair}")
        print(f"{'='*70}")
        print(f"Period: {period} | Interval: {interval}")
        print(f"ATR-based stops: {self.atr_sl_mult}x SL, {self.atr_tp_mult}x TP")
        print(f"Max Hold: {self.max_hold_time} minutes")
        print(f"{'='*70}\n")
        
        position = None
        entry_index = None
        relaxed_count = 0
        cooldown = {'long': 0, 'short': 0}
        
        for i in range(len(df)):
            current_time = df.index[i]
            
            # Decrement cooldown
            if cooldown['long'] > 0:
                cooldown['long'] -= 1
            if cooldown['short'] > 0:
                cooldown['short'] -= 1
            
            # Check exits if in position
            if position is not None:
                minutes_held = (i - entry_index)
                
                # TP exit
                if position['direction'] == 'long' and df.iloc[i]['Close'] >= position['tp_price']:
                    pnl = (df.iloc[i]['Close'] - position['entry_price']) / position['entry_price']
                    self.capital *= (1 + pnl)
                    print(f"✅ TAKE PROFIT ({position['direction'].upper()}): {df.iloc[i]['Close']:.5g} | PnL: {pnl*100:.2f}% | Held: {minutes_held}h | Capital: ${self.capital:.2f}")
                    self.trades.append({
                        'entry': position['entry_price'],
                        'exit': df.iloc[i]['Close'],
                        'direction': position['direction'],
                        'exit_type': 'TP',
                        'pnl': pnl,
                        'hold_time': minutes_held
                    })
                    position = None
                    cooldown['long'] = 10
                    continue
                
                if position['direction'] == 'short' and df.iloc[i]['Close'] <= position['tp_price']:
                    pnl = (position['entry_price'] - df.iloc[i]['Close']) / position['entry_price']
                    self.capital *= (1 + pnl)
                    print(f"✅ TAKE PROFIT ({position['direction'].upper()}): {df.iloc[i]['Close']:.5g} | PnL: {pnl*100:.2f}% | Held: {minutes_held}h | Capital: ${self.capital:.2f}")
                    self.trades.append({
                        'entry': position['entry_price'],
                        'exit': df.iloc[i]['Close'],
                        'direction': position['direction'],
                        'exit_type': 'TP',
                        'pnl': pnl,
                        'hold_time': minutes_held
                    })
                    position = None
                    cooldown['short'] = 10
                    continue
                
                # SL exit
                if position['direction'] == 'long' and df.iloc[i]['Close'] <= position['sl_price']:
                    pnl = (df.iloc[i]['Close'] - position['entry_price']) / position['entry_price']
                    self.capital *= (1 + pnl)
                    print(f"❌ STOP LOSS ({position['direction'].upper()}): {df.iloc[i]['Close']:.5g} | PnL: {pnl*100:.2f}% | Held: {minutes_held}h | Capital: ${self.capital:.2f}")
                    self.trades.append({
                        'entry': position['entry_price'],
                        'exit': df.iloc[i]['Close'],
                        'direction': position['direction'],
                        'exit_type': 'SL',
                        'pnl': pnl,
                        'hold_time': minutes_held
                    })
                    position = None
                    cooldown['long'] = 10
                    continue
                
                if position['direction'] == 'short' and df.iloc[i]['Close'] >= position['sl_price']:
                    pnl = (position['entry_price'] - df.iloc[i]['Close']) / position['entry_price']
                    self.capital *= (1 + pnl)
                    print(f"❌ STOP LOSS ({position['direction'].upper()}): {df.iloc[i]['Close']:.5g} | PnL: {pnl*100:.2f}% | Held: {minutes_held}h | Capital: ${self.capital:.2f}")
                    self.trades.append({
                        'entry': position['entry_price'],
                        'exit': df.iloc[i]['Close'],
                        'direction': position['direction'],
                        'exit_type': 'SL',
                        'pnl': pnl,
                        'hold_time': minutes_held
                    })
                    position = None
                    cooldown['short'] = 10
                    continue
                
                # Time exit
                if minutes_held >= self.max_hold_time / 4:  # 4h = 1 candle on 4h
                    pnl = (df.iloc[i]['Close'] - position['entry_price']) / position['entry_price'] if position['direction'] == 'long' else (position['entry_price'] - df.iloc[i]['Close']) / position['entry_price']
                    self.capital *= (1 + pnl)
                    print(f"⏰ TIME EXIT ({position['direction'].upper()}): {df.iloc[i]['Close']:.5g} | PnL: {pnl*100:.2f}% | Held: {minutes_held}h | Capital: ${self.capital:.2f}")
                    self.trades.append({
                        'entry': position['entry_price'],
                        'exit': df.iloc[i]['Close'],
                        'direction': position['direction'],
                        'exit_type': 'TIME',
                        'pnl': pnl,
                        'hold_time': minutes_held
                    })
                    position = None
                    continue
            
            # Generate entry signal
            if position is None:
                signal, sl_pct, tp_pct = self.generate_signal(df, i, relaxed=(relaxed_count > 200))
                
                if signal == 'buy' and cooldown['long'] == 0:
                    entry_price = df.iloc[i]['Close']
                    sl_price = entry_price * (1 - sl_pct)
                    tp_price = entry_price * (1 + tp_pct)
                    position = {
                        'direction': 'buy',
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price
                    }
                    entry_index = i
                    relaxed_count = 0
                    print(f"🟢 ENTER LONG: {entry_price:.5g} | SL: {sl_pct*100:.2f}% | TP: {tp_pct*100:.2f}% | RR: 1:{tp_pct/sl_pct:.1f}")
                
                elif signal == 'sell' and cooldown['short'] == 0:
                    entry_price = df.iloc[i]['Close']
                    sl_price = entry_price * (1 + sl_pct)
                    tp_price = entry_price * (1 - tp_pct)
                    position = {
                        'direction': 'sell',
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price
                    }
                    entry_index = i
                    relaxed_count = 0
                    print(f"🔴 ENTER SHORT: {entry_price:.5g} | SL: {sl_pct*100:.2f}% | TP: {tp_pct*100:.2f}% | RR: 1:{tp_pct/sl_pct:.1f}")
                else:
                    relaxed_count += 1
        
        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate backtest metrics."""
        if not self.trades:
            print(f"\n{'='*70}")
            print(f"NO TRADES EXECUTED")
            print(f"{'='*70}\n")
            return None
        
        trades_df = pd.DataFrame(self.trades)
        wins = (trades_df['pnl'] > 0).sum()
        losses = (trades_df['pnl'] <= 0).sum()
        win_rate = wins / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        total_pnl = (self.capital - self.starting_capital) / self.starting_capital * 100
        profit_trades = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        loss_trades = trades_df[trades_df['pnl'] <= 0]['pnl'].sum()
        profit_factor = abs(profit_trades / loss_trades) if loss_trades != 0 else float('inf')
        
        avg_hold = trades_df['hold_time'].mean()
        trades_per_day = len(trades_df) / 30  # 30-day period
        
        exit_breakdown = trades_df['exit_type'].value_counts()
        
        print(f"{'='*70}")
        print(f"HIGH WIN-RATE SCALPING RESULTS: {self.pair}")
        print(f"{'='*70}")
        print(f"Total Trades:     {len(trades_df)}")
        print(f"Wins:             {wins} ({win_rate:.1f}%)")
        print(f"Losses:           {losses}")
        print(f"Total Return:     ${self.capital - self.starting_capital:.2f} ({total_pnl:.2f}%)")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print(f"Avg Hold Time:    {avg_hold:.1f} candles (4h)")
        print(f"Trades/Day:       {trades_per_day:.1f}")
        print(f"Final Capital:    ${self.capital:.2f}\n")
        
        print(f"Exit Breakdown:")
        for exit_type, count in exit_breakdown.items():
            pct = count / len(trades_df) * 100
            print(f"  {exit_type:12} {count:3} ({pct:5.1f}%)")
        print(f"{'='*70}\n")
        
        target_wr = 55
        status = "✅" if win_rate >= target_wr else "❌"
        print(f"🎯 Target Win Rate: {target_wr}%")
        print(f"📊 Achieved: {win_rate:.1f}% {status}\n")
        
        return {
            'pair': self.pair,
            'trades': len(trades_df),
            'win_rate': win_rate,
            'roi': total_pnl,
            'profit_factor': profit_factor,
            'avg_hold': avg_hold,
            'trades_per_day': trades_per_day,
            'final_capital': self.capital,
            'exit_breakdown': exit_breakdown.to_dict()
        }


def test_aud_pairs_4h():
    """Test 4 best AUD pairs on 4h timeframe."""
    pairs = ['AUDNZD=X', 'AUDCAD=X', 'AUDHKD=X', 'AUDEUR=X']
    results = []
    
    print("\n" + "="*80)
    print(" "*20 + "4H TIMEFRAME TEST - TOP 4 AUD PAIRS")
    print("="*80 + "\n")
    
    for idx, pair in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] Testing {pair}...")
        bot = HighWRScalpingBot4H(pair, atr_sl_mult=1.4, atr_tp_mult=2.4, max_hold_minutes=240)
        result = bot.backtest(period='30d', interval='4h')
        if result:
            results.append(result)
    
    if results:
        print("\n" + "="*80)
        print("SUMMARY - 4H TIMEFRAME RESULTS")
        print("="*80)
        print(f"{'Pair':<15} {'Trades':>8} {'Win%':>8} {'ROI%':>8} {'TPD':>6} {'PF':>6} {'Status':<8}")
        print("-" * 80)
        
        total_trades = 0
        passed = 0
        
        for r in results:
            total_trades += r['trades']
            status = "✅ PASS" if r['win_rate'] >= 55 else "❌ FAIL"
            if r['win_rate'] >= 55:
                passed += 1
            print(f"{r['pair']:<15} {r['trades']:>8} {r['win_rate']:>7.1f}% {r['roi']:>7.2f}% {r['trades_per_day']:>6.1f} {r['profit_factor']:>6.2f} {status:<8}")
        
        print("-" * 80)
        avg_wr = sum(r['win_rate'] for r in results) / len(results)
        avg_roi = sum(r['roi'] for r in results) / len(results)
        print(f"{'AVERAGE':<15} {total_trades:>8} {avg_wr:>7.1f}% {avg_roi:>7.2f}%")
        print("="*80)
        print(f"\n✅ Pairs Passing (≥55% WR): {passed}/{len(results)}\n")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aud_4h_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to {filename}\n")
        
        return results
    else:
        print("\n❌ No results to display.\n")
        return []


if __name__ == "__main__":
    test_aud_pairs_4h()
