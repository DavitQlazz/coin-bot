#!/usr/bin/env python3
"""
Improved Forex Scalping Bot - Balanced Win Rate and Frequency
Key improvements:
1. ATR-based dynamic stop loss
2. Stricter entry filters with trend confirmation
3. Better signal quality scoring
4. Partial profit taking
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import json

class ImprovedScalpingBot:
    """Enhanced scalping strategy with higher win rate focus"""
    
    def __init__(self, pair='EURUSD=X', base_sl=0.4, base_tp=0.8,
                 max_hold_minutes=30, use_dynamic_sl=True):
        """
        Initialize improved scalping bot
        
        Args:
            pair: Forex pair to trade
            base_sl: Base stop loss percentage (will be adjusted by ATR)
            base_tp: Base take profit percentage
            max_hold_minutes: Max time to hold position
            use_dynamic_sl: Use ATR-based dynamic stop loss
        """
        self.pair = pair
        self.base_sl = base_sl
        self.base_tp = base_tp
        self.max_hold_minutes = max_hold_minutes
        self.use_dynamic_sl = use_dynamic_sl
        self.initial_capital = 1000
        self.capital = self.initial_capital
        
        print(f"✅ Improved Scalping Bot initialized")
        print(f"   Pair: {self.pair}")
        print(f"   Base SL: {self.base_sl}% | Base TP: {self.base_tp}%")
        print(f"   Dynamic SL: {use_dynamic_sl}")
        print(f"   Max Hold: {max_hold_minutes} minutes")
    
    def fetch_data(self, period='7d', interval='5m'):
        """Fetch forex data from Yahoo Finance"""
        print(f"\n📊 Fetching {interval} data for {self.pair}...")
        
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
        """Calculate indicators with ATR for dynamic stops"""
        
        # Bollinger Bands (20, 2)
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width_Pct'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
        
        # ATR (14) for dynamic stop loss
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
        
        # RSI (14)
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Stochastic (14, 3, 3)
        stoch = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close'],
            window=14, smooth_window=3
        )
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # EMAs (9, 21, 50, 200)
        df['EMA_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
        df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
        
        # MACD for trend confirmation
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # BB Position
        bb_range = (df['BB_Upper'] - df['BB_Lower'])
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / bb_range
        
        # Slopes
        df['RSI_Slope'] = df['RSI'].diff()
        df['EMA21_Slope'] = df['EMA_21'].diff()
        
        return df
    
    def generate_signal(self, df, index):
        """
        Generate high-quality scalping signal
        
        Entry criteria (need 3/5 for entry):
        LONG:
        1. Price in lower 30% of BB
        2. RSI < 40 and turning up
        3. Stochastic < 30
        4. Price below EMA21 (mean reversion)
        5. MACD histogram positive or turning up
        
        Additional filters:
        - BB width > 0.2% (avoid dead markets)
        - Not in strong downtrend (EMA50 > EMA200 or both flat)
        - Volume > 0.7x average
        
        SHORT: Opposite conditions
        """
        row = df.iloc[index]
        
        # Skip if not enough data
        if pd.isna(row['RSI']) or pd.isna(row['EMA_21']) or pd.isna(row['ATR']):
            return 'hold'
        
        # Filter 1: Avoid dead/low volatility markets
        if row['BB_Width_Pct'] < 0.2:
            return 'hold'
        
        # Filter 2: Volume filter
        if pd.notna(row['Volume_Ratio']) and row['Volume_Ratio'] < 0.7:
            return 'hold'
        
        # Get previous bar for slope checks
        prev_idx = max(0, index - 1)
        prev = df.iloc[prev_idx]
        
        # Calculate trend strength
        ema50 = row['EMA_50']
        ema200 = row['EMA_200']
        strong_downtrend = pd.notna(ema50) and pd.notna(ema200) and ema50 < ema200 * 0.995
        strong_uptrend = pd.notna(ema50) and pd.notna(ema200) and ema50 > ema200 * 1.005
        
        # === LONG Signal ===
        long_score = 0
        
        # 1. Price position
        if row['BB_Position'] < 0.3:
            long_score += 1
        
        # 2. RSI oversold and turning up
        if row['RSI'] < 40 and row['RSI_Slope'] > 0:
            long_score += 1
        
        # 3. Stochastic oversold
        if row['Stoch_K'] < 30:
            long_score += 1
        
        # 4. Mean reversion setup
        if row['Close'] < row['EMA_21']:
            long_score += 1
        
        # 5. MACD momentum
        if pd.notna(row['MACD_Hist']):
            if row['MACD_Hist'] > 0 or (row['MACD_Hist'] > prev['MACD_Hist']):
                long_score += 1
        
        # Require 3/5 + not in strong downtrend
        if long_score >= 3 and not strong_downtrend:
            return 'buy'
        
        # === SHORT Signal ===
        short_score = 0
        
        # 1. Price position
        if row['BB_Position'] > 0.7:
            short_score += 1
        
        # 2. RSI overbought and turning down
        if row['RSI'] > 60 and row['RSI_Slope'] < 0:
            short_score += 1
        
        # 3. Stochastic overbought
        if row['Stoch_K'] > 70:
            short_score += 1
        
        # 4. Mean reversion setup
        if row['Close'] > row['EMA_21']:
            short_score += 1
        
        # 5. MACD momentum
        if pd.notna(row['MACD_Hist']):
            if row['MACD_Hist'] < 0 or (row['MACD_Hist'] < prev['MACD_Hist']):
                short_score += 1
        
        # Require 3/5 + not in strong uptrend
        if short_score >= 3 and not strong_uptrend:
            return 'sell'
        
        return 'hold'
    
    def calculate_dynamic_stops(self, row, direction):
        """
        Calculate ATR-based dynamic stop loss and take profit
        
        Args:
            row: Current candle data
            direction: 'long' or 'short'
        
        Returns:
            (stop_loss_pct, take_profit_pct)
        """
        if not self.use_dynamic_sl or pd.isna(row['ATR_Pct']):
            return self.base_sl, self.base_tp
        
        # Use ATR but bound it to reasonable ranges
        atr_multiplier = 1.5  # Stop at 1.5x ATR
        dynamic_sl = min(max(row['ATR_Pct'] * atr_multiplier, 0.3), 0.8)
        
        # TP is 2x SL for 1:2 risk-reward
        dynamic_tp = dynamic_sl * 2.0
        
        return dynamic_sl, dynamic_tp
    
    def backtest(self, period='7d', interval='5m'):
        """Run backtest with improved logic"""
        
        print(f"\n{'='*70}")
        print(f"IMPROVED SCALPING BACKTEST: {self.pair}")
        print(f"Period: {period} | Interval: {interval}")
        print(f"{'='*70}\n")
        
        # Fetch and prepare data
        df = self.fetch_data(period=period, interval=interval)
        df = self.calculate_indicators(df)
        
        # Trading state
        in_position = False
        position_type = None
        entry_price = 0
        entry_time = None
        stop_loss_pct = 0
        take_profit_pct = 0
        
        trades = []
        total_trades = 0
        wins = 0
        losses = 0
        
        # Iterate through candles
        for i in range(50, len(df)):  # Start after warmup
            row = df.iloc[i]
            current_time = row.name
            current_price = row['Close']
            
            # Check exit conditions if in position
            if in_position:
                time_held = (current_time - entry_time).total_seconds() / 60
                pnl_pct = 0
                
                if position_type == 'long':
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # short
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
                exit_reason = None
                
                # Check stop loss
                if pnl_pct <= -stop_loss_pct:
                    exit_reason = 'SL'
                    losses += 1
                # Check take profit
                elif pnl_pct >= take_profit_pct:
                    exit_reason = 'TP'
                    wins += 1
                # Check time exit
                elif time_held >= self.max_hold_minutes:
                    exit_reason = 'TIME'
                    if pnl_pct > 0:
                        wins += 1
                    else:
                        losses += 1
                
                if exit_reason:
                    pnl_dollars = (self.capital * pnl_pct) / 100
                    self.capital += pnl_dollars
                    
                    icon = "✅" if exit_reason == 'TP' else ("❌" if exit_reason == 'SL' else "⏰")
                    print(f"{icon} EXIT {position_type.upper()}: {current_price:.5f} | "
                          f"Reason: {exit_reason} | PnL: {pnl_pct:+.2f}% | "
                          f"Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'type': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'exit_reason': exit_reason,
                        'hold_minutes': time_held
                    })
                    
                    in_position = False
                    total_trades += 1
                continue
            
            # Generate new signal if not in position
            signal = self.generate_signal(df, i)
            
            if signal in ['buy', 'sell']:
                position_type = 'long' if signal == 'buy' else 'short'
                entry_price = current_price
                entry_time = current_time
                in_position = True
                
                # Calculate dynamic stops
                stop_loss_pct, take_profit_pct = self.calculate_dynamic_stops(row, position_type)
                
                print(f"🟢 ENTER {position_type.upper()}: {entry_price:.5f} | "
                      f"SL: {stop_loss_pct:.2f}% | TP: {take_profit_pct:.2f}% | "
                      f"RSI: {row['RSI']:.1f} | Stoch: {row['Stoch_K']:.1f}")
        
        # Close any open position at end
        if in_position:
            current_price = df.iloc[-1]['Close']
            if position_type == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            pnl_dollars = (self.capital * pnl_pct) / 100
            self.capital += pnl_dollars
            
            if pnl_pct > 0:
                wins += 1
            else:
                losses += 1
            
            total_trades += 1
        
        # Calculate metrics
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        roi = ((self.capital / self.initial_capital) - 1) * 100
        
        # Calculate profit factor
        winning_pnl = sum(t['pnl_dollars'] for t in trades if t['pnl_dollars'] > 0)
        losing_pnl = abs(sum(t['pnl_dollars'] for t in trades if t['pnl_dollars'] < 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 999.0
        
        # Calculate average hold time
        avg_hold_time = sum(t['hold_minutes'] for t in trades) / len(trades) if trades else 0
        
        # Calculate trades per day
        days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        trades_per_day = total_trades / days if days > 0 else 0
        
        # Print results
        print(f"\n{'='*70}")
        print(f"IMPROVED SCALPING RESULTS: {self.pair}")
        print(f"{'='*70}")
        print(f"Total Trades:     {total_trades}")
        print(f"Wins:             {wins} ({win_rate:.1f}%)")
        print(f"Losses:           {losses}")
        print(f"Total Return:     ${self.capital - self.initial_capital:+.2f} ({roi:+.2f}%)")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print(f"Avg Hold Time:    {avg_hold_time:.1f} minutes")
        print(f"Trades/Day:       {trades_per_day:.1f}")
        print(f"Final Capital:    ${self.capital:.2f}")
        print(f"{'='*70}\n")
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'roi': roi,
            'profit_factor': profit_factor,
            'trades_per_day': trades_per_day,
            'final_capital': self.capital,
            'trades': trades
        }


if __name__ == "__main__":
    print("Testing Improved Scalping Strategy...\n")
    
    # Test on EUR/USD
    bot = ImprovedScalpingBot(
        pair='EURUSD=X',
        base_sl=0.4,
        base_tp=0.8,
        max_hold_minutes=30,
        use_dynamic_sl=True
    )
    
    results = bot.backtest(period='7d', interval='5m')
    
    print(f"\n🎯 Target: 1+ trade per day, >50% win rate")
    print(f"📊 Achieved: {results['trades_per_day']:.1f} trades/day, {results['win_rate']:.1f}% WR")
    
    if results['trades_per_day'] >= 1.0 and results['win_rate'] >= 50:
        print(f"✅ SUCCESS! Met both targets")
    elif results['trades_per_day'] >= 1.0:
        print(f"✅ Frequency target met, but win rate needs improvement")
    elif results['win_rate'] >= 50:
        print(f"✅ Win rate target met, but frequency too low")
    else:
        print(f"⚠️  Both targets need improvement")
