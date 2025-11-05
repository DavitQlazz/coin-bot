#!/usr/bin/env python3
"""
Enhanced Backtest with Real Binance Data
More aggressive strategy for better testing
"""

from backtest import Backtester
import os

class EnhancedBacktester(Backtester):
    def __init__(self):
        super().__init__()
        # More aggressive parameters for better signals
        self.stop_loss_pct = 3.0
        self.take_profit_pct = 6.0
    
    def generate_signal(self, df):
        """Enhanced signal generation with looser criteria"""
        if len(df) < 50:
            return 'HOLD'
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # More relaxed BUY conditions (need 2+ instead of 3+)
        buy_conditions = [
            latest['rsi'] < 40 and latest['rsi'] > previous['rsi'],  # Less strict RSI
            latest['macd'] > latest['macd_signal'],  # Simplified MACD
            latest['ema_9'] > latest['ema_21'],  # EMA trend
            latest['close'] < latest['bb_mid']  # Price position
        ]
        
        # SELL conditions
        sell_conditions = [
            latest['rsi'] > 65,  # Less strict overbought
            latest['macd'] < latest['macd_signal'],
            latest['ema_9'] < latest['ema_21'],
            latest['close'] > latest['bb_high']
        ]
        
        # Check stop loss and take profit
        if self.position:
            profit_pct = ((latest['close'] / self.position['entry_price']) - 1) * 100
            
            if profit_pct <= -self.stop_loss_pct:
                print(f"🛑 Stop Loss triggered: {profit_pct:.2f}%")
                return 'SELL'
            
            if profit_pct >= self.take_profit_pct:
                print(f"🎯 Take Profit triggered: {profit_pct:.2f}%")
                return 'SELL'
        
        # Generate signal (need 2+ conditions instead of 3+)
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        if buy_score >= 2 and not self.position:
            print(f"   📊 BUY signal: RSI={latest['rsi']:.1f}, MACD={'↑' if latest['macd'] > latest['macd_signal'] else '↓'}, Score={buy_score}/4")
            return 'BUY'
        elif (sell_score >= 2 or sell_conditions[0]) and self.position:
            print(f"   📊 SELL signal: RSI={latest['rsi']:.1f}, MACD={'↑' if latest['macd'] > latest['macd_signal'] else '↓'}, Score={sell_score}/4")
            return 'SELL'
        
        return 'HOLD'


def run_multiple_tests():
    """Run backtests with different configurations"""
    print("\n" + "="*70)
    print("🚀 ENHANCED BACKTESTING WITH BINANCE DATA")
    print("="*70)
    print("\nTesting multiple configurations...\n")
    
    configs = [
        {'symbol': 'BTC/USDT', 'timeframe': '1h', 'days': 30},
        {'symbol': 'ETH/USDT', 'timeframe': '1h', 'days': 30},
        {'symbol': 'BTC/USDT', 'timeframe': '4h', 'days': 30},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(configs)}: {config['symbol']} - {config['timeframe']}")
        print(f"{'='*70}")
        
        try:
            backtester = EnhancedBacktester()
            backtester.backtest(
                days=config['days'],
                symbol=config['symbol'],
                timeframe=config['timeframe']
            )
            
            if backtester.trades:
                total_profit = sum(t['profit'] for t in backtester.trades)
                win_rate = len([t for t in backtester.trades if t['profit'] > 0]) / len(backtester.trades) * 100
                roi = ((backtester.capital / backtester.initial_capital) - 1) * 100
                
                results.append({
                    'config': f"{config['symbol']} ({config['timeframe']})",
                    'trades': len(backtester.trades),
                    'win_rate': win_rate,
                    'profit': total_profit,
                    'roi': roi
                })
            else:
                results.append({
                    'config': f"{config['symbol']} ({config['timeframe']})",
                    'trades': 0,
                    'win_rate': 0,
                    'profit': 0,
                    'roi': 0
                })
        
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
            continue
    
    # Print summary
    if results:
        print("\n" + "="*70)
        print("📊 BACKTEST SUMMARY - ALL CONFIGURATIONS")
        print("="*70)
        
        for result in results:
            print(f"\n{result['config']}:")
            print(f"  Trades: {result['trades']}")
            if result['trades'] > 0:
                print(f"  Win Rate: {result['win_rate']:.1f}%")
                print(f"  Total P/L: ${result['profit']:+.2f}")
                print(f"  ROI: {result['roi']:+.2f}%")
            else:
                print(f"  Status: No trades triggered")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    print("\n🔬 TESTING STRATEGY WITH REAL BINANCE DATA")
    print("=" * 70)
    print("\nUsing Binance public API (no authentication required)")
    print("This will test the strategy on real historical market data\n")
    
    # Run single backtest with enhanced strategy
    print("Running enhanced backtest on BTC/USDT (1h timeframe)...")
    backtester = EnhancedBacktester()
    backtester.backtest(days=30)
    
    # Optionally run multiple tests
    print("\n\n" + "="*70)
    response = input("Run additional tests on other pairs/timeframes? (y/n): ").lower()
    if response == 'y':
        run_multiple_tests()
    
    print("\n✅ Backtesting complete!")
    print("\n💡 Tips for better results:")
    print("  • Try different timeframes (4h, 1d for less noise)")
    print("  • Adjust strategy parameters in bot.py")
    print("  • Use optimize.py to find best settings")
    print("  • Test on multiple symbols to find opportunities")
