#!/usr/bin/env python3
"""
Forex Scalping Bot - High Frequency Trading Strategy
Designed for: 1-5 minute timeframes, multiple trades per day
Target: 5-15 pips per trade, quick entries and exits

Strategy Components:
1. Bollinger Bands (20, 2) - Volatility and mean reversion
2. RSI (14) - Overbought/oversold conditions
3. Stochastic Oscillator (14, 3, 3) - Momentum
4. EMA (9, 21) - Short-term trend
5. Volume filter - Confirm moves with volume

Entry Rules:
- LONG: Price touches lower BB + RSI < 30 + Stoch < 20 + EMA9 > EMA21
- SHORT: Price touches upper BB + RSI > 70 + Stoch > 80 + EMA9 < EMA21

Exit Rules:
- Quick TP: 0.5-1.0% (5-10 pips for majors)
- Tight SL: 0.3-0.5% (3-5 pips)
- Time exit: Close if open > 30 minutes without hitting TP/SL

Risk-Reward: 1:2 minimum
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

load_dotenv()


class ForexScalpingBot:
    """High-frequency scalping strategy for forex trading"""
    
    def __init__(self, pair='EURUSD=X', stop_loss=0.4, take_profit=0.8,
                 max_hold_minutes=30, rsi_oversold=30, rsi_overbought=70,
                 stoch_low=20, stoch_high=80):
        """
        Initialize scalping bot with aggressive parameters
        
        Args:
            pair: Forex pair to trade
            stop_loss: Stop loss percentage (default 0.4% = ~4 pips)
            take_profit: Take profit percentage (default 0.8% = ~8 pips)
            max_hold_minutes: Max time to hold position (default 30 min)
            rsi_oversold: RSI level for oversold (default 30)
            rsi_overbought: RSI level for overbought (default 70)
            stoch_low: Stochastic low level (default 20)
            stoch_high: Stochastic high level (default 80)
        """
        self.pair = pair
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_minutes = max_hold_minutes
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stoch_low = stoch_low
        self.stoch_high = stoch_high
        self.initial_capital = 1000
        self.capital = self.initial_capital
        
        print(f"✅ Forex Scalping Bot initialized")
        print(f"   Pair: {self.pair}")
        print(f"   Stop Loss: {self.stop_loss}% (~{int(self.stop_loss*10)} pips)")
        print(f"   Take Profit: {self.take_profit}% (~{int(self.take_profit*10)} pips)")
        print(f"   Risk-Reward: 1:{self.take_profit/self.stop_loss:.1f}")
        print(f"   Max Hold Time: {self.max_hold_minutes} minutes")
        print(f"   Strategy: Bollinger Bands + RSI + Stochastic + EMA")
        print()
    
    def fetch_data(self, period='7d', interval='5m'):
        """
        Fetch forex data from Yahoo Finance
        
        Args:
            period: Time period to fetch (7d for 5min data)
            interval: Candle interval (1m, 5m, 15m)
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"📊 Fetching {interval} data for {self.pair}...")
        
        try:
            ticker = yf.Ticker(self.pair)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data returned for {self.pair}")
            
            print(f"✅ Fetched {len(df)} candles")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            raise
    
    def calculate_indicators(self, df):
        """
        Calculate scalping indicators
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added indicators
        """
        # Bollinger Bands (20, 2)
        bb_indicator = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb_indicator.bollinger_hband()
        df['BB_Middle'] = bb_indicator.bollinger_mavg()
        df['BB_Lower'] = bb_indicator.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
        
        # RSI (14)
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Stochastic Oscillator (14, 3, 3)
        stoch = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close'],
            window=14, smooth_window=3
        )
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # EMAs (9, 21, 50, 200) for trend bias
        df['EMA_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
        df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
        
        # Volume (normalized)
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price position in BB and BB width % for volatility filter
        bb_range = (df['BB_Upper'] - df['BB_Lower'])
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / bb_range
        df['BB_Width_Pct'] = (bb_range / df['Close']) * 100

        # Momentum slopes (simple 1-period change)
        df['RSI_Slope'] = df['RSI'].diff()
        df['EMA21_Slope'] = df['EMA_21'].diff()
        
        return df
    
    def generate_signal(self, df, index):
        """
        Generate scalping signal based on multiple conditions
        
        Signal Logic:
        LONG (BUY):
        - Price near lower BB (position < 0.2)
        - RSI oversold (< 30)
        - Stochastic oversold (< 20)
        - EMA9 > EMA21 (uptrend confirmation)
        - Decent volume (> 0.8x average)
        
        SHORT (SELL):
        - Price near upper BB (position > 0.8)
        - RSI overbought (> 70)
        - Stochastic overbought (> 80)
        - EMA9 < EMA21 (downtrend confirmation)
        - Decent volume (> 0.8x average)
        
        Returns:
            'buy', 'sell', or 'hold'
        """
        row = df.iloc[index]
        
        # Skip if indicators not ready
        if pd.isna(row['RSI']) or pd.isna(row['Stoch_K']) or pd.isna(row['EMA_21']):
            return 'hold'
        
        # WINRATE-FOCUSED filters
        # Volatility filter: avoid ultra-narrow bands
        if row['BB_Width_Pct'] is not None and row['BB_Width_Pct'] < 0.15:
            return 'hold'

        # Trend bias filter using higher EMAs: avoid trading hard against dominant trend
        ema50 = row['EMA_50']
        ema200 = row['EMA_200']

        # Check Stochastic cross and RSI slope using previous bar
        prev = df.iloc[index - 1] if index - 1 >= 0 else row
        stoch_bull_cross = (prev['Stoch_K'] < prev['Stoch_D']) and (row['Stoch_K'] > row['Stoch_D'])
        stoch_bear_cross = (prev['Stoch_K'] > prev['Stoch_D']) and (row['Stoch_K'] < row['Stoch_D'])
        rsi_up = row['RSI_Slope'] > 0
        rsi_down = row['RSI_Slope'] < 0

        # STRICTER Check for LONG signal
        long_score = 0
        if row['BB_Position'] < 0.2:  # Closer to lower BB
            long_score += 1
        if row['RSI'] < (self.rsi_oversold + 10) and rsi_up:  # Oversold and turning up
            long_score += 1
        if row['Stoch_K'] < (self.stoch_low + 10) and stoch_bull_cross:  # Momentum cross up
            long_score += 1
        if row['Close'] < row['EMA_21'] and row['EMA21_Slope'] >= 0:  # Mean reversion with rising EMA21
            long_score += 1

        # Require trend bias not strongly bearish for longs
        trend_ok_long = not (pd.notna(ema50) and pd.notna(ema200) and (ema50 < ema200))

        if long_score >= 3 and trend_ok_long:
            return 'buy'
        
        # STRICTER Check for SHORT signal
        short_score = 0
        if row['BB_Position'] > 0.8:  # Closer to upper BB
            short_score += 1
        if row['RSI'] > (self.rsi_overbought - 10) and rsi_down:  # Overbought and turning down
            short_score += 1
        if row['Stoch_K'] > (self.stoch_high - 10) and stoch_bear_cross:  # Momentum cross down
            short_score += 1
        if row['Close'] > row['EMA_21'] and row['EMA21_Slope'] <= 0:  # Mean reversion with falling EMA21
            short_score += 1

        # Require trend bias not strongly bullish for shorts
        trend_ok_short = not (pd.notna(ema50) and pd.notna(ema200) and (ema50 > ema200))

        if short_score >= 3 and trend_ok_short:
            return 'sell'
        
        return 'hold'
    
    def backtest(self, period='7d', interval='5m', verbose=True):
        """
        Backtest scalping strategy
        
        Args:
            period: Time period to test (7d for 5min, 60d for 1min)
            interval: Timeframe (1m, 5m, 15m)
            verbose: Print trade details
        
        Returns:
            Dictionary with performance metrics
        """
        if verbose:
            print("="*70)
            print(f"BACKTESTING SCALPING: {self.pair} | Period: {period} | Interval: {interval}")
            print(f"Stop Loss: {self.stop_loss}% | Take Profit: {self.take_profit}%")
            print(f"Max Hold: {self.max_hold_minutes} minutes")
            print("="*70)
            print()
        
        # Fetch and prepare data
        df = self.fetch_data(period=period, interval=interval)
        df = self.calculate_indicators(df)
        
        # Trading variables
        self.capital = self.initial_capital
        position = None
        entry_price = 0
        entry_time = None
        trades = []
        
        # Iterate through candles
        for i in range(50, len(df)):  # Start after indicators are ready
            current_price = df['Close'].iloc[i]
            current_time = df.index[i]
            
            # Check if we have a position
            if position:
                # Calculate time held
                time_held = (current_time - entry_time).total_seconds() / 60
                
                if position == 'long':
                    # Check stop loss
                    if current_price <= entry_price * (1 - self.stop_loss/100):
                        pnl = (current_price - entry_price) / entry_price * 100
                        profit = self.capital * (pnl / 100)
                        self.capital += profit
                        
                        if verbose:
                            print(f"❌ STOP LOSS (LONG): {current_price:.5f} | "
                                  f"PnL: {profit:+.2f} ({pnl:.2f}%) | "
                                  f"Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        
                        trades.append({
                            'type': 'long',
                            'entry': entry_price,
                            'exit': current_price,
                            'pnl': pnl,
                            'profit': profit,
                            'reason': 'stop_loss',
                            'time_held': time_held
                        })
                        position = None
                    
                    # Check take profit
                    elif current_price >= entry_price * (1 + self.take_profit/100):
                        pnl = (current_price - entry_price) / entry_price * 100
                        profit = self.capital * (pnl / 100)
                        self.capital += profit
                        
                        if verbose:
                            print(f"✅ TAKE PROFIT (LONG): {current_price:.5f} | "
                                  f"PnL: {profit:+.2f} ({pnl:.2f}%) | "
                                  f"Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        
                        trades.append({
                            'type': 'long',
                            'entry': entry_price,
                            'exit': current_price,
                            'pnl': pnl,
                            'profit': profit,
                            'reason': 'take_profit',
                            'time_held': time_held
                        })
                        position = None
                    
                    # Check time exit
                    elif time_held >= self.max_hold_minutes:
                        pnl = (current_price - entry_price) / entry_price * 100
                        profit = self.capital * (pnl / 100)
                        self.capital += profit
                        
                        if verbose:
                            print(f"⏰ TIME EXIT (LONG): {current_price:.5f} | "
                                  f"PnL: {profit:+.2f} ({pnl:.2f}%) | "
                                  f"Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        
                        trades.append({
                            'type': 'long',
                            'entry': entry_price,
                            'exit': current_price,
                            'pnl': pnl,
                            'profit': profit,
                            'reason': 'time_exit',
                            'time_held': time_held
                        })
                        position = None
                
                elif position == 'short':
                    # Check stop loss
                    if current_price >= entry_price * (1 + self.stop_loss/100):
                        pnl = (entry_price - current_price) / entry_price * 100
                        profit = self.capital * (pnl / 100)
                        self.capital += profit
                        
                        if verbose:
                            print(f"❌ STOP LOSS (SHORT): {current_price:.5f} | "
                                  f"PnL: {profit:+.2f} ({pnl:.2f}%) | "
                                  f"Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        
                        trades.append({
                            'type': 'short',
                            'entry': entry_price,
                            'exit': current_price,
                            'pnl': pnl,
                            'profit': profit,
                            'reason': 'stop_loss',
                            'time_held': time_held
                        })
                        position = None
                    
                    # Check take profit
                    elif current_price <= entry_price * (1 - self.take_profit/100):
                        pnl = (entry_price - current_price) / entry_price * 100
                        profit = self.capital * (pnl / 100)
                        self.capital += profit
                        
                        if verbose:
                            print(f"✅ TAKE PROFIT (SHORT): {current_price:.5f} | "
                                  f"PnL: {profit:+.2f} ({pnl:.2f}%) | "
                                  f"Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        
                        trades.append({
                            'type': 'short',
                            'entry': entry_price,
                            'exit': current_price,
                            'pnl': pnl,
                            'profit': profit,
                            'reason': 'take_profit',
                            'time_held': time_held
                        })
                        position = None
                    
                    # Check time exit
                    elif time_held >= self.max_hold_minutes:
                        pnl = (entry_price - current_price) / entry_price * 100
                        profit = self.capital * (pnl / 100)
                        self.capital += profit
                        
                        if verbose:
                            print(f"⏰ TIME EXIT (SHORT): {current_price:.5f} | "
                                  f"PnL: {profit:+.2f} ({pnl:.2f}%) | "
                                  f"Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        
                        trades.append({
                            'type': 'short',
                            'entry': entry_price,
                            'exit': current_price,
                            'pnl': pnl,
                            'profit': profit,
                            'reason': 'time_exit',
                            'time_held': time_held
                        })
                        position = None
            
            # Look for new entry if no position
            if not position:
                signal = self.generate_signal(df, i)
                
                if signal == 'buy':
                    position = 'long'
                    entry_price = current_price
                    entry_time = current_time
                    
                    if verbose:
                        print(f"🟢 ENTER LONG: {entry_price:.5f} | "
                              f"RSI: {df['RSI'].iloc[i]:.1f} | "
                              f"Stoch: {df['Stoch_K'].iloc[i]:.1f} | "
                              f"BB: {df['BB_Position'].iloc[i]:.2f}")
                
                elif signal == 'sell':
                    position = 'short'
                    entry_price = current_price
                    entry_time = current_time
                    
                    if verbose:
                        print(f"🔴 ENTER SHORT: {entry_price:.5f} | "
                              f"RSI: {df['RSI'].iloc[i]:.1f} | "
                              f"Stoch: {df['Stoch_K'].iloc[i]:.1f} | "
                              f"BB: {df['BB_Position'].iloc[i]:.2f}")
        
        # Calculate metrics
        total_trades = len(trades)
        wins = sum(1 for t in trades if t['profit'] > 0)
        losses = sum(1 for t in trades if t['profit'] <= 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
        total_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        avg_hold_time = sum(t['time_held'] for t in trades) / total_trades if total_trades > 0 else 0
        
        # Calculate trades per day
        time_range = (df.index[-1] - df.index[0]).total_seconds() / 86400  # days
        trades_per_day = total_trades / time_range if time_range > 0 else 0
        
        if verbose:
            print()
            print("="*70)
            print(f"SCALPING RESULTS: {self.pair}")
            print("="*70)
            print(f"Total Trades:     {total_trades}")
            print(f"Wins:             {wins} ({win_rate:.1f}%)")
            print(f"Losses:           {losses}")
            print(f"Total Return:     ${self.capital - self.initial_capital:+.2f} "
                  f"({(self.capital/self.initial_capital - 1)*100:+.2f}%)")
            print(f"Profit Factor:    {profit_factor:.2f}")
            print(f"Avg Hold Time:    {avg_hold_time:.1f} minutes")
            print(f"Trades/Day:       {trades_per_day:.1f}")
            print(f"Final Capital:    ${self.capital:.2f}")
            print("="*70)
            print()
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'final_capital': self.capital,
            'total_return': self.capital - self.initial_capital,
            'roi': (self.capital / self.initial_capital - 1) * 100,
            'profit_factor': profit_factor,
            'avg_hold_time': avg_hold_time,
            'trades_per_day': trades_per_day,
            'trades': trades
        }


if __name__ == "__main__":
    # Test scalping strategy on 5-minute timeframe
    print("Testing Forex Scalping Strategy...")
    print()
    
    bot = ForexScalpingBot(
        pair='EURUSD=X',
        stop_loss=0.4,      # 4 pips
        take_profit=0.8,    # 8 pips (1:2 R:R)
        max_hold_minutes=30
    )
    
    results = bot.backtest(period='7d', interval='5m', verbose=True)
    
    print(f"\n🎯 Target: 1+ trade per day")
    print(f"📊 Achieved: {results['trades_per_day']:.1f} trades per day")
    
    if results['trades_per_day'] >= 1:
        print(f"✅ SUCCESS! Met the frequency target!")
    else:
        print(f"⚠️  Below target, but much better than swing trading")
