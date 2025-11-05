#!/usr/bin/env python3
"""
Demo Script - Tests the bot with simulated data
Works without API keys or exchange connection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bot import CryptoTradingBot

class DemoBot(CryptoTradingBot):
    def __init__(self):
        # Override parent __init__ to skip exchange connection
        self.symbol = 'BTC/USDT'
        self.timeframe = '1h'
        self.capital = 1000.0
        self.initial_capital = self.capital
        self.position = None
        self.trades = []
        self.paper_trading = True
        self.stop_loss_pct = 2.0
        self.take_profit_pct = 5.0
        self.exchange = None  # No exchange needed
    
    def generate_demo_data(self, days=30):
        """Generate realistic demo OHLCV data"""
        print("📊 Generating demo market data...\n")
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1H')
        
        # Generate realistic price movements
        base_price = 45000  # Starting BTC price
        volatility = 0.02   # 2% volatility
        trend = 0.0001      # Slight upward trend
        
        prices = [base_price]
        for i in range(len(timestamps) - 1):
            # Random walk with drift
            change = np.random.randn() * volatility + trend
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'close': prices
        })
        
        # Generate high/low with realistic spreads
        df['high'] = df['close'] * (1 + np.abs(np.random.randn(len(df)) * 0.01))
        df['low'] = df['close'] * (1 - np.abs(np.random.randn(len(df)) * 0.01))
        
        # Add volume
        df['volume'] = np.random.uniform(100, 1000, len(df))
        
        return df
    
    def run_demo(self, days=30):
        """Run a demo backtest with generated data"""
        print(f"{'='*60}")
        print(f"🎮 DEMO MODE - Simulated Trading Bot")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Period: {days} days")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"{'='*60}\n")
        
        # Generate demo data
        df = self.generate_demo_data(days=days)
        
        # Calculate indicators
        print("🔄 Calculating technical indicators...")
        df = self.calculate_indicators(df)
        
        print(f"✅ Ready! Running strategy on {len(df)} candles\n")
        print(f"{'='*60}\n")
        
        # Run backtest
        trade_count = 0
        for i in range(50, len(df)):
            window = df.iloc[:i+1]
            signal = self.generate_signal(window)
            current_price = window.iloc[-1]['close']
            timestamp = window.iloc[-1]['timestamp']
            
            # Execute trade
            if signal != 'HOLD':
                self.execute_trade(signal, current_price)
                if signal == 'BUY' or (signal == 'SELL' and self.trades):
                    trade_count += 1
            
            # Show progress every 100 candles
            if i % 100 == 0:
                print(f"📍 Progress: {i}/{len(df)} candles processed...")
        
        # Close any open position
        if self.position:
            print("\n⚠️  Closing final position...")
            final_price = df.iloc[-1]['close']
            self.execute_trade('SELL', final_price)
        
        # Show results
        self.print_summary()
        self.show_demo_insights(df)
    
    def show_demo_insights(self, df):
        """Show additional insights about the demo"""
        print("\n" + "="*60)
        print("💡 DEMO INSIGHTS")
        print("="*60)
        print("\nThis was a simulated demo using randomly generated data.")
        print("\nTo use the bot with real data:")
        print("  1. Get API keys from your exchange (Binance, Coinbase, etc.)")
        print("  2. Add keys to .env file")
        print("  3. Run: python backtest.py (safe backtesting)")
        print("  4. Run: python bot.py (live trading)")
        
        print("\n⚠️  IMPORTANT REMINDERS:")
        print("  • Always backtest before live trading")
        print("  • Start with paper trading (PAPER_TRADING=true)")
        print("  • Never risk more than you can afford to lose")
        print("  • Crypto markets are highly volatile")
        
        print("\n📚 Next Steps:")
        print("  • Optimize strategy parameters: python optimize.py")
        print("  • Monitor markets: python monitor.py")
        print("  • Customize strategy in bot.py")
        
        print("="*60)


if __name__ == "__main__":
    print("\n🎮 Starting Trading Bot Demo\n")
    print("This demo simulates trading without requiring API keys.")
    print("It generates realistic market data and tests the strategy.\n")
    
    demo = DemoBot()
    demo.run_demo(days=30)
    
    print("\n✅ Demo complete! Review the results above.\n")
