#!/usr/bin/env python3
"""
Strategy Optimizer
Tests different parameter combinations to find optimal settings
"""

from bot import CryptoTradingBot
from backtest import Backtester
import itertools
import os

class StrategyOptimizer:
    def __init__(self):
        self.results = []
    
    def optimize(self, days=30):
        """Test different strategy parameters"""
        print("\n" + "="*60)
        print("🔧 STRATEGY OPTIMIZER")
        print("="*60)
        
        # Parameter ranges to test
        rsi_thresholds = [(30, 70), (25, 75), (35, 65)]
        ema_pairs = [(9, 21), (12, 26), (5, 20)]
        stop_loss_values = [1.5, 2.0, 3.0]
        take_profit_values = [3.0, 5.0, 7.0]
        
        total_combinations = (len(rsi_thresholds) * len(ema_pairs) * 
                            len(stop_loss_values) * len(take_profit_values))
        
        print(f"Testing {total_combinations} parameter combinations...")
        print(f"This may take a while...\n")
        
        combination_num = 0
        
        for rsi, ema, sl, tp in itertools.product(rsi_thresholds, ema_pairs, 
                                                   stop_loss_values, take_profit_values):
            combination_num += 1
            
            # Set parameters
            os.environ['STOP_LOSS_PERCENT'] = str(sl)
            os.environ['TAKE_PROFIT_PERCENT'] = str(tp)
            
            print(f"\n[{combination_num}/{total_combinations}] Testing: "
                  f"RSI({rsi[0]},{rsi[1]}) EMA({ema[0]},{ema[1]}) "
                  f"SL:{sl}% TP:{tp}%")
            
            try:
                # Run backtest with these parameters
                backtester = Backtester()
                backtester.backtest(days=days)
                
                if backtester.trades:
                    total_profit = sum(t['profit'] for t in backtester.trades)
                    win_rate = (len([t for t in backtester.trades if t['profit'] > 0]) / 
                              len(backtester.trades) * 100)
                    roi = ((backtester.capital / backtester.initial_capital) - 1) * 100
                    
                    self.results.append({
                        'rsi': rsi,
                        'ema': ema,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'total_trades': len(backtester.trades),
                        'win_rate': win_rate,
                        'total_profit': total_profit,
                        'roi': roi,
                        'final_capital': backtester.capital
                    })
                    
                    print(f"  → Profit: ${total_profit:+.2f} | ROI: {roi:+.2f}% | "
                          f"Win Rate: {win_rate:.1f}% | Trades: {len(backtester.trades)}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        # Print top results
        self.print_top_results()
    
    def print_top_results(self):
        """Print the best performing parameter combinations"""
        if not self.results:
            print("\n❌ No successful backtests completed")
            return
        
        print("\n" + "="*60)
        print("🏆 TOP PERFORMING STRATEGIES")
        print("="*60)
        
        # Sort by ROI
        sorted_by_roi = sorted(self.results, key=lambda x: x['roi'], reverse=True)[:5]
        
        print("\n📈 Best by ROI:")
        print("-" * 60)
        for i, result in enumerate(sorted_by_roi, 1):
            print(f"{i}. ROI: {result['roi']:+.2f}% | Profit: ${result['total_profit']:+.2f} | "
                  f"Win Rate: {result['win_rate']:.1f}%")
            print(f"   RSI: {result['rsi']} | EMA: {result['ema']} | "
                  f"SL: {result['stop_loss']}% | TP: {result['take_profit']}%")
            print()
        
        # Sort by win rate
        sorted_by_winrate = sorted(self.results, key=lambda x: x['win_rate'], reverse=True)[:5]
        
        print("\n🎯 Best by Win Rate:")
        print("-" * 60)
        for i, result in enumerate(sorted_by_winrate, 1):
            print(f"{i}. Win Rate: {result['win_rate']:.1f}% | ROI: {result['roi']:+.2f}% | "
                  f"Trades: {result['total_trades']}")
            print(f"   RSI: {result['rsi']} | EMA: {result['ema']} | "
                  f"SL: {result['stop_loss']}% | TP: {result['take_profit']}%")
            print()
        
        print("="*60)


if __name__ == "__main__":
    optimizer = StrategyOptimizer()
    optimizer.optimize(days=30)
