import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import ta

class ForexTradingBotV2:
    """
    Enhanced Forex Trading Bot V2 with improved strategy
    
    Key Improvements:
    1. ADX (Average Directional Index) - Only trade strong trends
    2. Multi-condition signal scoring - Require stronger confluence
    3. Trend strength filtering - Avoid ranging markets
    4. Dynamic position sizing - Based on signal confidence
    5. Better risk management - Trailing stops on winners
    """
    
    def __init__(self):
        load_dotenv()
        self.symbol = os.getenv('FOREX_PAIR', 'EURUSD=X')
        self.interval = os.getenv('FOREX_INTERVAL', '1h')
        self.capital = float(os.getenv('INITIAL_CAPITAL', 10000))
        self.initial_capital = self.capital
        self.position = None
        self.trades = []
        self.paper_trading = True
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PERCENT', 3.0))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PERCENT', 6.0))
        
        # V2 Enhanced parameters
        self.min_adx = float(os.getenv('MIN_ADX', 25))  # Minimum trend strength
        self.min_signal_score = float(os.getenv('MIN_SIGNAL_SCORE', 3))  # Minimum conditions
        self.use_trailing_stop = os.getenv('USE_TRAILING_STOP', 'true').lower() == 'true'
        self.trailing_stop_trigger = 0.03  # Start trailing after 3% profit
        self.trailing_stop_distance = 0.015  # Trail 1.5% from peak
        
        print(f"✅ Enhanced Forex Trading Bot V2 initialized")
        print(f"   Pair: {self.symbol}")
        print(f"   Interval: {self.interval}")
        print(f"   Capital: ${self.capital:,.2f}")
        print(f"   Min ADX: {self.min_adx} (trend strength filter)")
        print(f"   Min Signal Score: {self.min_signal_score}")
        print(f"   Trailing Stop: {self.use_trailing_stop}")
        
    def fetch_ohlcv(self, period='30d'):
        """Fetch historical forex data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period, interval=self.interval)
            
            if df.empty:
                print(f"❌ No data fetched for {self.symbol}")
                return None
            
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_cols]
            
            print(f"✅ Fetched {len(df)} candles for {self.symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators including ADX"""
        # RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']  # Volatility measure
        
        # EMAs (Exponential Moving Averages)
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        
        # ATR (Average True Range) for volatility
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # ADX (Average Directional Index) - KEY ADDITION for trend strength
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()  # Positive directional indicator
        df['adx_neg'] = adx.adx_neg()  # Negative directional indicator
        
        # Stochastic Oscillator - Additional momentum indicator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def calculate_trend_strength(self, df):
        """Calculate overall trend strength and direction"""
        latest = df.iloc[-1]
        
        # ADX interpretation:
        # ADX < 20: No trend (ranging market)
        # ADX 20-25: Weak trend
        # ADX 25-50: Strong trend
        # ADX > 50: Very strong trend
        
        adx_value = latest['adx']
        
        if adx_value < 20:
            return 'RANGING', adx_value
        elif latest['adx_pos'] > latest['adx_neg']:
            return 'UPTREND', adx_value
        else:
            return 'DOWNTREND', adx_value
    
    def generate_signal(self, df):
        """
        Enhanced signal generation with scoring system
        Returns: ('BUY'|'SELL'|'HOLD', confidence_score)
        """
        if len(df) < 200:
            return 'HOLD', 0
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # First check: ADX trend strength filter
        trend_direction, adx_strength = self.calculate_trend_strength(df)
        
        if adx_strength < self.min_adx:
            # Market is ranging, don't trade
            return 'HOLD', 0
        
        # Calculate signal scores (0-7 points possible)
        buy_score = 0
        sell_score = 0
        
        # 1. RSI Conditions (1 point)
        if latest['rsi'] < 40 and latest['rsi'] > previous['rsi']:
            buy_score += 1
        if latest['rsi'] > 60 and latest['rsi'] < previous['rsi']:
            sell_score += 1
        
        # 2. MACD Conditions (1 point)
        if latest['macd'] > latest['macd_signal'] and previous['macd'] <= previous['macd_signal']:
            buy_score += 1
        if latest['macd'] < latest['macd_signal'] and previous['macd'] >= previous['macd_signal']:
            sell_score += 1
        
        # 3. EMA Trend Alignment (1 point)
        if latest['ema_9'] > latest['ema_21'] > latest['ema_50']:
            buy_score += 1
        if latest['ema_9'] < latest['ema_21'] < latest['ema_50']:
            sell_score += 1
        
        # 4. Price vs EMA 200 (1 point) - Major trend
        if latest['close'] > latest['ema_200']:
            buy_score += 1
        if latest['close'] < latest['ema_200']:
            sell_score += 1
        
        # 5. Bollinger Band Position (1 point)
        if latest['close'] < latest['bb_low'] * 1.01:
            buy_score += 1
        if latest['close'] > latest['bb_high'] * 0.99:
            sell_score += 1
        
        # 6. Stochastic Oscillator (1 point)
        if latest['stoch_k'] < 20 and latest['stoch_k'] > previous['stoch_k']:
            buy_score += 1
        if latest['stoch_k'] > 80 and latest['stoch_k'] < previous['stoch_k']:
            sell_score += 1
        
        # 7. ADX Direction Confirmation (1 point)
        if trend_direction == 'UPTREND':
            buy_score += 1
        if trend_direction == 'DOWNTREND':
            sell_score += 1
        
        # Decision logic: Need minimum score AND trend confirmation
        if buy_score >= self.min_signal_score and trend_direction == 'UPTREND':
            return 'BUY', buy_score
        elif sell_score >= self.min_signal_score and trend_direction == 'DOWNTREND':
            return 'SELL', sell_score
        else:
            return 'HOLD', max(buy_score, sell_score)
    
    def execute_trade(self, signal, confidence_score, current_price, timestamp):
        """Execute trade with dynamic position sizing based on confidence"""
        if signal == 'BUY' and self.position is None:
            # Dynamic position sizing: Higher confidence = larger position
            base_size = 0.95
            confidence_multiplier = min(confidence_score / 7.0, 1.0)  # Max 100%
            position_size = self.capital * base_size * (0.7 + 0.3 * confidence_multiplier)  # 70-100%
            
            units = position_size / current_price
            
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'units': units,
                'entry_time': timestamp,
                'stop_loss': current_price * (1 - self.stop_loss_pct / 100),
                'take_profit': current_price * (1 + self.take_profit_pct / 100),
                'confidence': confidence_score,
                'peak_price': current_price,  # For trailing stop
                'trailing_active': False
            }
            
            print(f"🟢 BUY: {units:.4f} units @ {current_price:.5f} | Confidence: {confidence_score}/7 | Size: {position_size/self.capital*100:.1f}%")
            print(f"   Stop: {self.position['stop_loss']:.5f} | Target: {self.position['take_profit']:.5f}")
            
        elif signal == 'SELL' and self.position is None:
            base_size = 0.95
            confidence_multiplier = min(confidence_score / 7.0, 1.0)
            position_size = self.capital * base_size * (0.7 + 0.3 * confidence_multiplier)
            
            units = position_size / current_price
            
            self.position = {
                'type': 'SHORT',
                'entry_price': current_price,
                'units': units,
                'entry_time': timestamp,
                'stop_loss': current_price * (1 + self.stop_loss_pct / 100),
                'take_profit': current_price * (1 - self.take_profit_pct / 100),
                'confidence': confidence_score,
                'peak_price': current_price,  # For trailing stop (will track lowest for SHORT)
                'trailing_active': False
            }
            
            print(f"🔴 SELL: {units:.4f} units @ {current_price:.5f} | Confidence: {confidence_score}/7 | Size: {position_size/self.capital*100:.1f}%")
            print(f"   Stop: {self.position['stop_loss']:.5f} | Target: {self.position['take_profit']:.5f}")
    
    def update_trailing_stop(self, current_price):
        """Update trailing stop for winning positions"""
        if self.position is None or not self.use_trailing_stop:
            return
        
        entry_price = self.position['entry_price']
        
        if self.position['type'] == 'LONG':
            profit_pct = (current_price - entry_price) / entry_price
            
            # Update peak price
            if current_price > self.position['peak_price']:
                self.position['peak_price'] = current_price
            
            # Activate trailing stop after trigger threshold
            if profit_pct >= self.trailing_stop_trigger:
                if not self.position['trailing_active']:
                    self.position['trailing_active'] = True
                    print(f"   🎯 Trailing stop activated at {profit_pct*100:.2f}% profit")
                
                # Update stop loss to trail peak by distance
                new_stop = self.position['peak_price'] * (1 - self.trailing_stop_distance)
                if new_stop > self.position['stop_loss']:
                    self.position['stop_loss'] = new_stop
                    
        elif self.position['type'] == 'SHORT':
            profit_pct = (entry_price - current_price) / entry_price
            
            # Update lowest price (peak for SHORT)
            if current_price < self.position['peak_price']:
                self.position['peak_price'] = current_price
            
            # Activate trailing stop after trigger threshold
            if profit_pct >= self.trailing_stop_trigger:
                if not self.position['trailing_active']:
                    self.position['trailing_active'] = True
                    print(f"   🎯 Trailing stop activated at {profit_pct*100:.2f}% profit")
                
                # Update stop loss to trail lowest by distance
                new_stop = self.position['peak_price'] * (1 + self.trailing_stop_distance)
                if new_stop < self.position['stop_loss']:
                    self.position['stop_loss'] = new_stop
    
    def check_exit_conditions(self, current_price, timestamp):
        """Check exit conditions including trailing stops"""
        if self.position is None:
            return
        
        # Update trailing stop first
        self.update_trailing_stop(current_price)
        
        should_exit = False
        exit_reason = ""
        
        if self.position['type'] == 'LONG':
            if current_price <= self.position['stop_loss']:
                should_exit = True
                exit_reason = "Trailing Stop" if self.position['trailing_active'] else "Stop Loss"
            elif current_price >= self.position['take_profit']:
                should_exit = True
                exit_reason = "Take Profit"
                
        elif self.position['type'] == 'SHORT':
            if current_price >= self.position['stop_loss']:
                should_exit = True
                exit_reason = "Trailing Stop" if self.position['trailing_active'] else "Stop Loss"
            elif current_price <= self.position['take_profit']:
                should_exit = True
                exit_reason = "Take Profit"
        
        if should_exit:
            self.close_position(current_price, timestamp, exit_reason)
    
    def close_position(self, current_price, timestamp, reason="Manual"):
        """Close position and record trade"""
        if self.position is None:
            return
        
        entry_price = self.position['entry_price']
        units = self.position['units']
        
        if self.position['type'] == 'LONG':
            pnl = (current_price - entry_price) * units
        else:  # SHORT
            pnl = (entry_price - current_price) * units
        
        self.capital += pnl
        pnl_pct = (pnl / (entry_price * units)) * 100
        
        trade = {
            'entry_time': str(self.position['entry_time']),
            'exit_time': str(timestamp),
            'type': self.position['type'],
            'entry_price': entry_price,
            'exit_price': current_price,
            'units': units,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'confidence': self.position['confidence'],
            'capital_after': self.capital
        }
        
        self.trades.append(trade)
        
        emoji = "✅" if pnl > 0 else "❌"
        print(f"{emoji} CLOSE {self.position['type']}: {units:.4f} units @ {current_price:.5f} | "
              f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | Reason: {reason} | "
              f"Capital: ${self.capital:,.2f}")
        
        self.position = None
    
    def save_trades(self, filename='trades_v2.json'):
        """Save trade history to file"""
        with open(filename, 'w') as f:
            json.dump({
                'initial_capital': self.initial_capital,
                'final_capital': self.capital,
                'total_return': self.capital - self.initial_capital,
                'return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
                'total_trades': len(self.trades),
                'trades': self.trades
            }, f, indent=2)
        print(f"✅ Trades saved to {filename}")
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return None
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        total_return = self.capital - self.initial_capital
        return_pct = (total_return / self.initial_capital) * 100
        
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        
        total_wins = sum([t['pnl'] for t in wins])
        total_losses = abs(sum([t['pnl'] for t in losses]))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_return': total_return,
            'return_pct': return_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': self.capital
        }
