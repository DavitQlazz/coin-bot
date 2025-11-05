#!/usr/bin/env python3
"""
Real-time Market Monitor
Displays live market data and signals without executing trades
"""

from bot import CryptoTradingBot
import time
from datetime import datetime
import os

class MarketMonitor(CryptoTradingBot):
    def __init__(self):
        super().__init__()
        self.paper_trading = True
    
    def monitor(self):
        """Monitor market without trading"""
        print(f"\n{'='*80}")
        print(f"📊 MARKET MONITOR - {self.symbol}")
        print(f"{'='*80}\n")
        
        try:
            while True:
                df = self.fetch_ohlcv(limit=100)
                
                if df is None:
                    print("⚠️  Failed to fetch data, retrying...")
                    time.sleep(30)
                    continue
                
                df = self.calculate_indicators(df)
                latest = df.iloc[-1]
                signal = self.generate_signal(df)
                
                # Clear screen for clean display (optional)
                # os.system('clear' if os.name == 'posix' else 'cls')
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"\n{'='*80}")
                print(f"⏰ {timestamp}")
                print(f"{'='*80}")
                
                print(f"\n💹 PRICE DATA")
                print(f"-" * 80)
                print(f"Current Price: ${latest['close']:,.2f}")
                print(f"24h High: ${df['high'].tail(24).max():,.2f}")
                print(f"24h Low: ${df['low'].tail(24).min():,.2f}")
                print(f"24h Change: {((latest['close'] / df.iloc[-25]['close']) - 1) * 100:+.2f}%")
                
                print(f"\n📊 TECHNICAL INDICATORS")
                print(f"-" * 80)
                print(f"RSI (14): {latest['rsi']:.2f} {'🔥 Overbought' if latest['rsi'] > 70 else '❄️  Oversold' if latest['rsi'] < 30 else '✅ Neutral'}")
                print(f"MACD: {latest['macd']:.2f}")
                print(f"MACD Signal: {latest['macd_signal']:.2f}")
                print(f"MACD Histogram: {latest['macd_diff']:.2f} {'📈' if latest['macd_diff'] > 0 else '📉'}")
                print(f"\nEMA 9: ${latest['ema_9']:,.2f}")
                print(f"EMA 21: ${latest['ema_21']:,.2f}")
                print(f"EMA 50: ${latest['ema_50']:,.2f}")
                print(f"Trend: {'📈 Bullish' if latest['ema_9'] > latest['ema_21'] else '📉 Bearish'}")
                
                print(f"\n🎯 BOLLINGER BANDS")
                print(f"-" * 80)
                print(f"Upper Band: ${latest['bb_high']:,.2f}")
                print(f"Middle Band: ${latest['bb_mid']:,.2f}")
                print(f"Lower Band: ${latest['bb_low']:,.2f}")
                bb_position = ((latest['close'] - latest['bb_low']) / 
                              (latest['bb_high'] - latest['bb_low']) * 100)
                print(f"Position: {bb_position:.1f}% {'⬆️  Upper' if bb_position > 80 else '⬇️  Lower' if bb_position < 20 else '➡️  Middle'}")
                
                print(f"\n🎬 TRADING SIGNAL")
                print(f"-" * 80)
                signal_icon = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
                print(f"{signal_icon} {signal}")
                
                print(f"\n{'='*80}")
                
                # Wait before next update
                time.sleep(60)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Monitor stopped by user")


if __name__ == "__main__":
    monitor = MarketMonitor()
    monitor.monitor()
