"""
Forex Parameter Optimizer
Find optimal trading parameters for forex pairs using Yahoo Finance data
"""

import itertools
import json
from datetime import datetime
from backtest_forex import ForexBacktester
import os

class ForexOptimizer:
    """Optimize forex trading parameters across multiple pairs"""
    
    def __init__(self):
        self.results = []
        
    def test_configuration(self, pair, stop_loss, take_profit, period='90d', interval='4h'):
        """Test a single parameter configuration"""
        try:
            # Update environment variables temporarily
            os.environ['STOP_LOSS_PERCENT'] = str(stop_loss)
            os.environ['TAKE_PROFIT_PERCENT'] = str(take_profit)
            
            # Create backtester with new config
            backtester = ForexBacktester(pair=pair)
            backtester.stop_loss_pct = stop_loss
            backtester.take_profit_pct = take_profit
            
            # Run backtest
            metrics = backtester.backtest(period=period, interval=interval)
            
            if metrics and metrics['total_trades'] >= 1:  # Need minimum 1 trade
                return {
                    'pair': pair,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'period': period,
                    'interval': interval,
                    **metrics
                }
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error testing {pair} SL={stop_loss}% TP={take_profit}%: {e}")
            return None
    
    def optimize_pair(self, pair, period='90d', interval='4h'):
        """Optimize parameters for a single pair"""
        print(f"\n{'='*80}")
        print(f"🔍 OPTIMIZING: {pair}")
        print(f"{'='*80}")
        print(f"Period: {period}, Interval: {interval}")
        
        # Parameter ranges - fewer configs for faster testing
        stop_loss_values = [2.5, 3.0, 3.5, 4.0]
        take_profit_values = [5.0, 6.0, 7.0, 8.0]
        
        total_configs = len(stop_loss_values) * len(take_profit_values)
        print(f"Testing {total_configs} configurations...\n")
        
        pair_results = []
        config_num = 0
        
        for sl, tp in itertools.product(stop_loss_values, take_profit_values):
            config_num += 1
            print(f"[{config_num}/{total_configs}] Testing SL={sl}% TP={tp}%...", end=' ')
            
            result = self.test_configuration(pair, sl, tp, period, interval)
            
            if result:
                pair_results.append(result)
                print(f"✅ ROI: {result['roi']:+.2f}%, Win Rate: {result['win_rate']:.1f}%, Trades: {result['total_trades']}")
            else:
                print(f"⚠️  Insufficient trades")
        
        return pair_results
    
    def optimize_all_pairs(self, pairs, period='60d', interval='1h'):
        """Optimize parameters for multiple pairs"""
        print("\n" + "="*80)
        print(f"{'FOREX PARAMETER OPTIMIZATION':^80}")
        print("="*80)
        print(f"Testing {len(pairs)} pairs with {period} of {interval} data")
        print("="*80 + "\n")
        
        all_results = []
        
        for pair in pairs:
            pair_results = self.optimize_pair(pair, period, interval)
            all_results.extend(pair_results)
        
        self.results = all_results
        return all_results
    
    def analyze_results(self):
        """Analyze optimization results and find best configurations"""
        if not self.results:
            print("❌ No results to analyze")
            return None
        
        print("\n" + "="*80)
        print(f"{'OPTIMIZATION RESULTS ANALYSIS':^80}")
        print("="*80)
        
        # Overall statistics
        profitable = [r for r in self.results if r['roi'] > 0]
        total_configs = len(self.results)
        
        print(f"\nTotal Configurations Tested: {total_configs}")
        print(f"Profitable Configurations: {len(profitable)} ({len(profitable)/total_configs*100:.1f}%)")
        print(f"Average ROI: {sum(r['roi'] for r in self.results)/total_configs:.2f}%")
        print(f"Average Win Rate: {sum(r['win_rate'] for r in self.results)/total_configs:.2f}%")
        
        # Best by ROI
        best_roi = max(self.results, key=lambda x: x['roi'])
        print(f"\n🏆 BEST ROI:")
        print(f"   Pair: {best_roi['pair']}")
        print(f"   Stop Loss: {best_roi['stop_loss']}%")
        print(f"   Take Profit: {best_roi['take_profit']}%")
        print(f"   ROI: {best_roi['roi']:+.2f}%")
        print(f"   Win Rate: {best_roi['win_rate']:.2f}%")
        print(f"   Trades: {best_roi['total_trades']}")
        print(f"   Profit Factor: {best_roi['profit_factor']:.2f}")
        
        # Best by win rate
        best_wr = max(self.results, key=lambda x: x['win_rate'])
        print(f"\n🎯 BEST WIN RATE:")
        print(f"   Pair: {best_wr['pair']}")
        print(f"   Stop Loss: {best_wr['stop_loss']}%")
        print(f"   Take Profit: {best_wr['take_profit']}%")
        print(f"   Win Rate: {best_wr['win_rate']:.2f}%")
        print(f"   ROI: {best_wr['roi']:+.2f}%")
        print(f"   Trades: {best_wr['total_trades']}")
        
        # Best by profit factor (among profitable)
        if profitable:
            best_pf = max(profitable, key=lambda x: x['profit_factor'])
            print(f"\n💰 BEST PROFIT FACTOR:")
            print(f"   Pair: {best_pf['pair']}")
            print(f"   Stop Loss: {best_pf['stop_loss']}%")
            print(f"   Take Profit: {best_pf['take_profit']}%")
            print(f"   Profit Factor: {best_pf['profit_factor']:.2f}")
            print(f"   ROI: {best_pf['roi']:+.2f}%")
            print(f"   Win Rate: {best_pf['win_rate']:.2f}%")
        
        # Best configuration by composite score
        for r in self.results:
            # Composite score: 50% ROI, 30% Win Rate, 20% Profit Factor
            r['composite_score'] = (
                r['roi'] * 0.5 +
                r['win_rate'] * 0.3 +
                min(r['profit_factor'], 3.0) * 10 * 0.2  # Cap profit factor at 3
            )
        
        best_overall = max(self.results, key=lambda x: x['composite_score'])
        print(f"\n⭐ BEST OVERALL (Composite Score):")
        print(f"   Pair: {best_overall['pair']}")
        print(f"   Stop Loss: {best_overall['stop_loss']}%")
        print(f"   Take Profit: {best_overall['take_profit']}%")
        print(f"   ROI: {best_overall['roi']:+.2f}%")
        print(f"   Win Rate: {best_overall['win_rate']:.2f}%")
        print(f"   Profit Factor: {best_overall['profit_factor']:.2f}")
        print(f"   Trades: {best_overall['total_trades']}")
        print(f"   Composite Score: {best_overall['composite_score']:.2f}")
        
        # Top 10 configurations
        print(f"\n{'='*80}")
        print(f"{'TOP 10 CONFIGURATIONS':^80}")
        print(f"{'='*80}\n")
        
        top_10 = sorted(self.results, key=lambda x: x['composite_score'], reverse=True)[:10]
        
        print(f"{'Rank':<5} {'Pair':<12} {'SL%':<6} {'TP%':<6} {'ROI%':<8} {'Win%':<8} {'PF':<8} {'Trades':<8} {'Score':<8}")
        print("-"*80)
        
        for i, r in enumerate(top_10, 1):
            print(f"{i:<5} {r['pair']:<12} {r['stop_loss']:<6.1f} {r['take_profit']:<6.1f} "
                  f"{r['roi']:<8.2f} {r['win_rate']:<8.1f} {r['profit_factor']:<8.2f} "
                  f"{r['total_trades']:<8} {r['composite_score']:<8.2f}")
        
        print("\n" + "="*80)
        
        # Parameter recommendations
        print(f"\n{'='*80}")
        print(f"{'PARAMETER RECOMMENDATIONS':^80}")
        print(f"{'='*80}\n")
        
        # Find most common parameters in top 10
        if profitable:
            top_profitable = sorted(profitable, key=lambda x: x['composite_score'], reverse=True)[:5]
            avg_sl = sum(r['stop_loss'] for r in top_profitable) / len(top_profitable)
            avg_tp = sum(r['take_profit'] for r in top_profitable) / len(top_profitable)
            
            print(f"Based on top profitable configurations:")
            print(f"  Recommended Stop Loss: {avg_sl:.1f}%")
            print(f"  Recommended Take Profit: {avg_tp:.1f}%")
            print(f"  Risk/Reward Ratio: 1:{avg_tp/avg_sl:.2f}")
        else:
            print("No profitable configurations found.")
            print("Recommendation: Test longer trending periods or different pairs")
        
        print("\n" + "="*80)
        
        return {
            'best_roi': best_roi,
            'best_win_rate': best_wr,
            'best_overall': best_overall,
            'top_10': top_10
        }
    
    def save_results(self, filename=None):
        """Save optimization results to JSON"""
        if not filename:
            filename = f"forex_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename


def run_optimization():
    """Run comprehensive forex optimization"""
    print("\n" + "="*80)
    print(f"{'FOREX PARAMETER OPTIMIZATION':^80}")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Major forex pairs
    pairs = [
        'EURUSD=X',
        'GBPUSD=X',
        'USDJPY=X',
        'AUDUSD=X',
    ]
    
    # Create optimizer
    optimizer = ForexOptimizer()
    
    # Run optimization with 90 days of 4-hour data for better signals
    print("Using 90 days of 4-hour data for optimization...")
    print("(4-hour timeframe provides smoother signals and better trends)\n")
    results = optimizer.optimize_all_pairs(pairs, period='90d', interval='4h')
    
    if results:
        # Analyze results
        analysis = optimizer.analyze_results()
        
        # Save results
        optimizer.save_results()
        
        return analysis
    else:
        print("❌ No valid results from optimization")
        return None


def quick_test():
    """Quick test with fewer configurations"""
    print("\n" + "="*80)
    print(f"{'QUICK FOREX OPTIMIZATION TEST':^80}")
    print("="*80 + "\n")
    
    pairs = ['EURUSD=X', 'USDJPY=X']
    
    optimizer = ForexOptimizer()
    
    # Test just a few key configurations
    print("Testing key configurations on 2 pairs...\n")
    
    configs = [
        (2.5, 5.0),
        (3.0, 6.0),
        (3.5, 7.0),
    ]
    
    for pair in pairs:
        print(f"\n{'='*80}")
        print(f"Testing {pair}")
        print(f"{'='*80}\n")
        
        for sl, tp in configs:
            print(f"Testing SL={sl}% TP={tp}%...", end=' ')
            result = optimizer.test_configuration(pair, sl, tp, '60d', '1h')
            
            if result:
                optimizer.results.append(result)
                print(f"✅ ROI: {result['roi']:+.2f}%, Win Rate: {result['win_rate']:.1f}%")
            else:
                print(f"⚠️  Insufficient trades")
    
    if optimizer.results:
        optimizer.analyze_results()
        optimizer.save_results()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_test()
    else:
        run_optimization()
