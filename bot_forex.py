import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import ta

class ForexTradingBot:
    """
    Forex Trading Bot using Yahoo Finance API
    Supports major forex pairs: EUR/USD, GBP/USD, USD/JPY, AUD/USD, etc.
    """
    
    def __init__(self):
        load_dotenv()
        self.symbol = os.getenv('FOREX_PAIR', 'EURUSD=X')  # Yahoo format
        self.interval = os.getenv('FOREX_INTERVAL', '1h')  # 1m, 5m, 15m, 1h, 1d
        self.capital = float(os.getenv('INITIAL_CAPITAL', 10000))
        self.initial_capital = self.capital
        self.position = None
        self.trades = []
        self.paper_trading = True  # Always paper trade for forex
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PERCENT', 3.0))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PERCENT', 6.0))
        
        print(f"✅ Forex Trading Bot initialized")
        print(f"   Pair: {self.symbol}")
        print(f"   Interval: {self.interval}")
        print(f"   Capital: ${self.capital:,.2f}")
        
    def fetch_ohlcv(self, period='30d'):
        """
        Fetch historical forex data from Yahoo Finance
        
        Args:
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period, interval=self.interval)
            
            if df.empty:
                print(f"❌ No data fetched for {self.symbol}")
                return None
            
            # Rename columns to match crypto bot format
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            
            # Ensure we have the right columns
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            
            # Keep only required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_cols]
            
            print(f"✅ Fetched {len(df)} candles for {self.symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for forex"""
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
        
        # EMA (Exponential Moving Average)
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        # ATR (Average True Range) for volatility
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        return df
    
    def generate_signal(self, df):
        """Generate trading signals based on multiple indicators"""
        if len(df) < 50:
            return 'HOLD'
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # BUY conditions (optimized for forex)
        buy_conditions = [
            latest['rsi'] < 35 and latest['rsi'] > previous['rsi'],  # RSI recovering from oversold
            latest['macd'] > latest['macd_signal'] and previous['macd'] <= previous['macd_signal'],  # MACD crossover
            latest['ema_9'] > latest['ema_21'],  # Short EMA above medium
            latest['close'] < latest['bb_low'] * 1.02,  # Near lower Bollinger Band
        ]
        
        # SELL conditions
        sell_conditions = [
            latest['rsi'] > 70 and latest['rsi'] < previous['rsi'],  # RSI declining from overbought
            latest['macd'] < latest['macd_signal'] and previous['macd'] >= previous['macd_signal'],  # MACD bearish cross
            latest['ema_9'] < latest['ema_21'],  # Short EMA below medium
            latest['close'] > latest['bb_high'] * 0.98,  # Near upper Bollinger Band
        ]
        
        # Require at least 2 conditions to be met (optimized setting)
        if sum(buy_conditions) >= 2:
            return 'BUY'
        elif sum(sell_conditions) >= 2:
            return 'SELL'
        else:
            return 'HOLD'
    
    def execute_trade(self, signal, current_price, timestamp):
        """Execute paper trading for forex"""
        if signal == 'BUY' and self.position is None:
            # Open long position
            position_size = self.capital * 0.95  # Use 95% of capital
            units = position_size / current_price
            
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'units': units,
                'entry_time': timestamp,
                'stop_loss': current_price * (1 - self.stop_loss_pct / 100),
                'take_profit': current_price * (1 + self.take_profit_pct / 100)
            }
            
            print(f"🟢 BUY: {units:.4f} units @ {current_price:.5f} | Stop: {self.position['stop_loss']:.5f} | Target: {self.position['take_profit']:.5f}")
            
        elif signal == 'SELL' and self.position is None:
            # Open short position
            position_size = self.capital * 0.95
            units = position_size / current_price
            
            self.position = {
                'type': 'SHORT',
                'entry_price': current_price,
                'units': units,
                'entry_time': timestamp,
                'stop_loss': current_price * (1 + self.stop_loss_pct / 100),
                'take_profit': current_price * (1 - self.take_profit_pct / 100)
            }
            
            print(f"🔴 SELL: {units:.4f} units @ {current_price:.5f} | Stop: {self.position['stop_loss']:.5f} | Target: {self.position['take_profit']:.5f}")
    
    def check_exit_conditions(self, current_price, timestamp):
        """Check if we should exit current position"""
        if self.position is None:
            return
        
        should_exit = False
        exit_reason = ""
        
        if self.position['type'] == 'LONG':
            # Check stop loss and take profit for LONG
            if current_price <= self.position['stop_loss']:
                should_exit = True
                exit_reason = "Stop Loss"
            elif current_price >= self.position['take_profit']:
                should_exit = True
                exit_reason = "Take Profit"
                
        elif self.position['type'] == 'SHORT':
            # Check stop loss and take profit for SHORT
            if current_price >= self.position['stop_loss']:
                should_exit = True
                exit_reason = "Stop Loss"
            elif current_price <= self.position['take_profit']:
                should_exit = True
                exit_reason = "Take Profit"
        
        if should_exit:
            self.close_position(current_price, timestamp, exit_reason)
    
    def close_position(self, current_price, timestamp, reason="Manual"):
        """Close current position and record trade"""
        if self.position is None:
            return
        
        entry_price = self.position['entry_price']
        units = self.position['units']
        
        if self.position['type'] == 'LONG':
            pnl = (current_price - entry_price) * units
        else:  # SHORT
            pnl = (entry_price - current_price) * units
        
        pnl_pct = (pnl / (entry_price * units)) * 100
        
        self.capital += pnl
        
        trade_record = {
            'type': self.position['type'],
            'entry_price': entry_price,
            'exit_price': current_price,
            'units': units,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_time': str(self.position['entry_time']),
            'exit_time': str(timestamp),
            'reason': reason
        }
        
        self.trades.append(trade_record)
        
        print(f"{'🟢' if pnl > 0 else '🔴'} CLOSE {self.position['type']}: {units:.4f} units @ {current_price:.5f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
        
        self.position = None
    
    def calculate_metrics(self):
        """Calculate trading performance metrics"""
        if not self.trades:
            return None
        
        df_trades = pd.DataFrame(self.trades)
        
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        total_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        net_profit = self.capital - self.initial_capital
        roi = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_profit': net_profit,
            'roi': roi,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_capital': self.capital
        }
    
    def print_summary(self):
        """Print trading summary"""
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("\n📊 No trades executed")
            return
        
        print("\n" + "="*70)
        print(f"{'FOREX TRADING SUMMARY':^70}")
        print("="*70)
        print(f"Pair: {self.symbol}")
        print(f"Interval: {self.interval}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"Net Profit: ${metrics['net_profit']:,.2f}")
        print(f"ROI: {metrics['roi']:.2f}%")
        print("-"*70)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Average Win: ${metrics['avg_win']:.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:.2f}")
        print("="*70)
    
    def run(self, period='30d'):
        """Run the forex trading bot"""
        print(f"\n🤖 Starting Forex Trading Bot for {self.symbol}")
        print(f"⏰ Period: {period}")
        
        # Fetch data
        df = self.fetch_ohlcv(period=period)
        if df is None or df.empty:
            return
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Simulate trading through historical data
        print(f"\n📈 Running strategy simulation...")
        for i in range(50, len(df)):  # Start after warm-up period
            current_data = df.iloc[:i+1]
            current_price = current_data.iloc[-1]['close']
            current_time = current_data.iloc[-1]['timestamp']
            
            # Check exit conditions first
            self.check_exit_conditions(current_price, current_time)
            
            # Generate new signal
            signal = self.generate_signal(current_data)
            
            # Execute trade
            self.execute_trade(signal, current_price, current_time)
        
        # Close any open position
        if self.position:
            self.close_position(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], "End of period")
        
        # Print summary
        self.print_summary()
        
        return self.calculate_metrics()

if __name__ == '__main__':
    bot = ForexTradingBot()
    bot.run()
