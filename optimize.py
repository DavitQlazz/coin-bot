#!/usr/bin/env python3
"""
Advanced Strategy Optimizer Using Real Binance Data
Automatically finds optimal parameters based on historical performance
"""

import ccxt
from backtest import Backtester
import itertools
from datetime import datetime
import json

class AdvancedOptimizer:
    def __init__(self):
        self.results = []
        self.best_config = None
        
    def test_configuration(self, config):
        """Test a specific configuration"""
        class CustomBacktester(Backtester):
            def __init__(self, params):
                super().__init__()
                self.stop_loss_pct = params['stop_loss']
                self.take_profit_pct = params['take_profit']
                self.rsi_oversold = params['rsi_oversold']
                self.rsi_overbought = params['rsi_overbought']
                self.min_conditions = params['min_conditions']
                
            def generate_signal(self, df):
                """Custom signal generation with configurable parameters"""
                if len(df) < 50:
                    return 'HOLD'
                
                latest = df.iloc[-1]
                previous = df.iloc[-2]
                
                # BUY conditions with configurable RSI threshold
                buy_conditions = [
                    latest['rsi'] < self.rsi_oversold and latest['rsi'] > previous['rsi'],
                    latest['macd'] > latest['macd_signal'],
                    latest['ema_9'] > latest['ema_21'],
                    latest['close'] < latest['bb_mid']
                ]
                
                # SELL conditions with configurable RSI threshold
                sell_conditions = [
                    latest['rsi'] > self.rsi_overbought,
                    latest['macd'] < latest['macd_signal'],
                    latest['ema_9'] < latest['ema_21'],
                    latest['close'] > latest['bb_high']
                ]
                
                # Check stop loss and take profit
                if self.position:
                    profit_pct = ((latest['close'] / self.position['entry_price']) - 1) * 100
                    
                    if profit_pct <= -self.stop_loss_pct:
                        return 'SELL'
                    
                    if profit_pct >= self.take_profit_pct:
                        return 'SELL'
                
                # Generate signal with configurable minimum conditions
                buy_score = sum(buy_conditions)
                sell_score = sum(sell_conditions)
                
                if buy_score >= self.min_conditions and not self.position:
                    return 'BUY'
                elif (sell_score >= self.min_conditions or sell_conditions[0]) and self.position:
                    return 'SELL'
                
                return 'HOLD'
        
        try:
            backtester = CustomBacktester(config)
            # Suppress output during optimization
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            backtester.backtest(days=config['days'], symbol=config['symbol'], timeframe=config['timeframe'])
            
            sys.stdout = old_stdout
            
            if backtester.trades and len(backtester.trades) >= 5:  # Need at least 5 trades
                total_profit = sum(t['profit'] for t in backtester.trades)
                winning_trades = [t for t in backtester.trades if t['profit'] > 0]
                win_rate = len(winning_trades) / len(backtester.trades) * 100
                roi = ((backtester.capital / backtester.initial_capital) - 1) * 100
                
                # Calculate profit factor
                total_wins = sum(t['profit'] for t in backtester.trades if t['profit'] > 0)
                total_losses = abs(sum(t['profit'] for t in backtester.trades if t['profit'] < 0))
                profit_factor = total_wins / total_losses if total_losses > 0 else 0
                
                return {
                    'config': config,
                    'trades': len(backtester.trades),
                    'win_rate': win_rate,
                    'profit': total_profit,
                    'roi': roi,
                    'profit_factor': profit_factor,
                    'avg_profit_per_trade': total_profit / len(backtester.trades)
                }
            
            return None
            
        except Exception as e:
            return None
    
    def optimize(self, symbol='BTC/USDT', timeframe='1h', days=30):
        """Run optimization with multiple parameter combinations"""
        print("\n" + "="*70)
        print("🔧 ADVANCED STRATEGY OPTIMIZER")
        print("="*70)
        print(f"\nSymbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Period: {days} days")
        print(f"Data Source: Binance Public API")
        print("\nOptimizing parameters based on real market data...")
        print("="*70 + "\n")
        
        # Parameter ranges to test (optimized based on previous results)
        param_grid = {
            'stop_loss': [2.0, 2.5, 3.0, 3.5],
            'take_profit': [4.0, 5.0, 6.0, 7.0],
            'rsi_oversold': [30, 35, 40],
            'rsi_overbought': [65, 70, 75],
            'min_conditions': [2, 3]
        }
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        total = len(combinations)
        print(f"Testing {total} parameter combinations...\n")
        
        completed = 0
        for i, params in enumerate(combinations, 1):
            params['symbol'] = symbol
            params['timeframe'] = timeframe
            params['days'] = days
            
            print(f"[{i}/{total}] Testing: SL={params['stop_loss']}%, TP={params['take_profit']}%, "
                  f"RSI=({params['rsi_oversold']},{params['rsi_overbought']}), "
                  f"MinCond={params['min_conditions']}", end='')
            
            result = self.test_configuration(params)
            
            if result:
                self.results.append(result)
                print(f" → ROI: {result['roi']:+.2f}%, Win: {result['win_rate']:.1f}%, "
                      f"PF: {result['profit_factor']:.2f}, Trades: {result['trades']}")
                completed += 1
            else:
                print(" → Skip")
            
            # Show progress every 24 tests
            if i % 24 == 0:
                print(f"\n  Progress: {i}/{total} ({i/total*100:.1f}%) - {completed} valid results\n")
        
        print("\n" + "="*70)
        print(f"✅ Optimization complete! Tested {completed} valid configurations")
        print("="*70)
        
        if not self.results:
            print("\n❌ No valid results found. Try:")
            print("  • Longer test period (60+ days)")
            print("  • Different timeframe (4h or 1d)")
            print("  • Different trading pair")
            return
        
        # Analyze results
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze and display optimization results"""
        if not self.results:
            return
        
        print("\n" + "="*70)
        print("📊 OPTIMIZATION RESULTS ANALYSIS")
        print("="*70)
        
        # Calculate composite score (balanced approach)
        for r in self.results:
            # Composite score considers ROI, win rate, profit factor, and trade count
            r['score'] = (
                r['roi'] * 0.4 +  # 40% weight on ROI
                r['win_rate'] * 0.3 +  # 30% weight on win rate
                r['profit_factor'] * 10 +  # 20% weight on profit factor (scaled)
                min(r['trades'] / 50 * 10, 10)  # 10% weight on trade count (capped)
            )
        
        by_score = sorted(self.results, key=lambda x: x['score'], reverse=True)
        by_roi = sorted(self.results, key=lambda x: x['roi'], reverse=True)
        by_winrate = sorted(self.results, key=lambda x: x['win_rate'], reverse=True)
        by_profit_factor = sorted(self.results, key=lambda x: x['profit_factor'], reverse=True)
        
        # Display top results by composite score
        print("\n🏆 TOP 5 CONFIGURATIONS (RECOMMENDED)")
        print("-" * 70)
        for i, r in enumerate(by_score[:5], 1):
            c = r['config']
            print(f"\n{i}. Score: {r['score']:.2f}")
            print(f"   Parameters:")
            print(f"     • Stop Loss: {c['stop_loss']}%")
            print(f"     • Take Profit: {c['take_profit']}%")
            print(f"     • RSI Range: {c['rsi_oversold']}-{c['rsi_overbought']}")
            print(f"     • Min Conditions: {c['min_conditions']}")
            print(f"   Performance:")
            print(f"     • ROI: {r['roi']:+.2f}%")
            print(f"     • Win Rate: {r['win_rate']:.1f}%")
            print(f"     • Profit Factor: {r['profit_factor']:.2f}")
            print(f"     • Total Trades: {r['trades']}")
            print(f"     • Avg Profit/Trade: ${r['avg_profit_per_trade']:+.2f}")
        
        # Save best configuration
        self.best_config = by_score[0]['config']
        
        print("\n📈 BEST BY SPECIFIC METRICS:")
        print("-" * 70)
        
        print(f"\n💰 Highest ROI: {by_roi[0]['roi']:+.2f}%")
        c = by_roi[0]['config']
        print(f"   SL={c['stop_loss']}%, TP={c['take_profit']}%, "
              f"RSI=({c['rsi_oversold']},{c['rsi_overbought']}), MinCond={c['min_conditions']}")
        
        print(f"\n🎯 Highest Win Rate: {by_winrate[0]['win_rate']:.1f}%")
        c = by_winrate[0]['config']
        print(f"   SL={c['stop_loss']}%, TP={c['take_profit']}%, "
              f"RSI=({c['rsi_oversold']},{c['rsi_overbought']}), MinCond={c['min_conditions']}")
        
        print(f"\n⚖️  Best Profit Factor: {by_profit_factor[0]['profit_factor']:.2f}")
        c = by_profit_factor[0]['config']
        print(f"   SL={c['stop_loss']}%, TP={c['take_profit']}%, "
              f"RSI=({c['rsi_oversold']},{c['rsi_overbought']}), MinCond={c['min_conditions']}")
        
        # Statistical analysis
        print("\n\n📊 STATISTICAL ANALYSIS:")
        print("-" * 70)
        avg_roi = sum(r['roi'] for r in self.results) / len(self.results)
        avg_winrate = sum(r['win_rate'] for r in self.results) / len(self.results)
        avg_pf = sum(r['profit_factor'] for r in self.results) / len(self.results)
        
        profitable = len([r for r in self.results if r['roi'] > 0])
        
        print(f"Total Configurations Tested: {len(self.results)}")
        print(f"Profitable Configurations: {profitable} ({profitable/len(self.results)*100:.1f}%)")
        print(f"Average ROI: {avg_roi:+.2f}%")
        print(f"Average Win Rate: {avg_winrate:.1f}%")
        print(f"Average Profit Factor: {avg_pf:.2f}")
        
        # Save results
        self.save_results()
        
        print("\n" + "="*70)
        print("💡 RECOMMENDATIONS:")
        print("="*70)
        print(f"\nBest configuration saved to: optimization_results.json")
        print("\nTo use the optimized settings, update your .env file:")
        c = self.best_config
        print(f"""
STOP_LOSS_PERCENT={c['stop_loss']}
TAKE_PROFIT_PERCENT={c['take_profit']}

# Then modify bot.py generate_signal() to use:
# RSI oversold: {c['rsi_oversold']}
# RSI overbought: {c['rsi_overbought']}
# Minimum conditions: {c['min_conditions']}
""")
        
        print("\nNext steps:")
        print("  1. Apply these settings to your bot")
        print("  2. Run more backtests to confirm: python backtest.py")
        print("  3. Test with paper trading: python bot.py")
        print("  4. Monitor performance and adjust as needed")
        
        print("\n⚠️  Important:")
        print("  • Past performance doesn't guarantee future results")
        print("  • Market conditions change constantly")
        print("  • Re-optimize parameters regularly (weekly/monthly)")
        print("  • Always test with paper trading first")
        print("="*70)
    
    def save_results(self):
        """Save optimization results to file"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'best_config': self.best_config,
                'top_10_results': sorted(self.results, key=lambda x: x['score'], reverse=True)[:10],
                'statistics': {
                    'total_tested': len(self.results),
                    'profitable': len([r for r in self.results if r['roi'] > 0]),
                    'avg_roi': sum(r['roi'] for r in self.results) / len(self.results),
                    'avg_win_rate': sum(r['win_rate'] for r in self.results) / len(self.results)
                }
            }
            
            with open('optimization_results.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: optimization_results.json")
            
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")


def quick_optimize():
    """Quick optimization with reduced parameter space"""
    print("\n🚀 QUICK OPTIMIZATION MODE")
    print("Testing most promising parameter ranges...\n")
    
    optimizer = AdvancedOptimizer()
    
    # Quick test with reduced combinations (32 total)
    param_grid = {
        'stop_loss': [2.5, 3.0],
        'take_profit': [5.0, 6.0],
        'rsi_oversold': [35, 40],
        'rsi_overbought': [65, 70],
        'min_conditions': [2, 3]
    }
    
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(combinations)} quick combinations on 4h timeframe...\n")
    
    for i, params in enumerate(combinations, 1):
        params['symbol'] = 'BTC/USDT'
        params['timeframe'] = '4h'  # Use 4h for faster/more reliable results
        params['days'] = 30
        
        print(f"[{i}/{len(combinations)}] Testing: SL={params['stop_loss']}%, TP={params['take_profit']}%, "
              f"RSI=({params['rsi_oversold']},{params['rsi_overbought']}), MinCond={params['min_conditions']}", end='')
        
        result = optimizer.test_configuration(params)
        if result:
            optimizer.results.append(result)
            print(f" → ROI: {result['roi']:+.2f}%, Trades: {result['trades']}")
        else:
            print(" → Skip")
    
    if optimizer.results:
        optimizer.analyze_results()
    else:
        print("\n❌ No valid results. Try full optimization with longer period.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 STRATEGY OPTIMIZATION WITH REAL BINANCE DATA")
    print("="*70)
    print("\nThis will test multiple parameter combinations on real market data")
    print("to find the most profitable settings for your trading bot.\n")
    
    print("Choose optimization mode:")
    print("  1. Quick Optimization (~32 combinations, ~3-5 minutes)")
    print("  2. Full Optimization (~96 combinations, ~10-15 minutes)")
    print("  3. Custom Symbol/Timeframe")
    print()
    
    choice = input("Enter choice (1-3) [1]: ").strip() or "1"
    
    if choice == "1":
        quick_optimize()
    elif choice == "2":
        optimizer = AdvancedOptimizer()
        optimizer.optimize(symbol='BTC/USDT', timeframe='4h', days=30)
    elif choice == "3":
        symbol = input("Enter symbol [BTC/USDT]: ").strip() or "BTC/USDT"
        timeframe = input("Enter timeframe [4h]: ").strip() or "4h"
        days = int(input("Enter days [30]: ").strip() or "30")
        
        optimizer = AdvancedOptimizer()
        optimizer.optimize(symbol=symbol, timeframe=timeframe, days=days)
    else:
        print("Invalid choice. Running quick optimization...")
        quick_optimize()
    
    print("\n✅ Optimization complete!")
