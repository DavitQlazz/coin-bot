#!/usr/bin/env python3
"""
Test different timeframes for scalping strategy
"""

from bot_forex_scalping import ForexScalpingBot
from datetime import datetime
import json

# Best performing pairs
BEST_PAIRS = ['GBPUSD=X', 'USDJPY=X']

# Timeframes to test
TIMEFRAME_CONFIGS = [
    {'interval': '1m', 'period': '5d', 'name': '1-minute (ultra high-freq)'},
    {'interval': '5m', 'period': '7d', 'name': '5-minute (high-freq)'},
    {'interval': '15m', 'period': '30d', 'name': '15-minute (medium-freq)'},
    {'interval': '30m', 'period': '60d', 'name': '30-minute (medium)'},
    {'interval': '60m', 'period': '90d', 'name': '1-hour (lower-freq)'},
]

# Best SL/TP from optimization
OPTIMAL_SL = 0.3
OPTIMAL_TP = 0.6

def test_timeframe(pair, interval, period, config_name):
    """Test a single timeframe"""
    print(f"\nTesting {config_name} on {pair}...")
    try:
        bot = ForexScalpingBot(
            pair=pair,
            stop_loss=OPTIMAL_SL,
            take_profit=OPTIMAL_TP,
            max_hold_minutes=30
        )
        
        results = bot.backtest(period=period, interval=interval)
        
        return {
            'pair': pair,
            'config': config_name,
            'interval': interval,
            'period': period,
            'trades': results['total_trades'],
            'wins': results['wins'],
            'win_rate': results['win_rate'],
            'roi': results['roi'],
            'profit_factor': results['profit_factor'],
            'trades_per_day': results['trades_per_day']
        }
    except Exception as e:
        print(f"❌ Error testing {pair} on {interval}: {e}")
        return None

def main():
    print("="*70)
    print("SCALPING STRATEGY - TIMEFRAME OPTIMIZATION")
    print("="*70)
    print(f"Testing {len(TIMEFRAME_CONFIGS)} timeframes on {len(BEST_PAIRS)} pairs")
    print(f"Optimal SL: {OPTIMAL_SL}% | Optimal TP: {OPTIMAL_TP}%")
    print("="*70)
    
    all_results = []
    
    for pair in BEST_PAIRS:
        print(f"\n{'='*70}")
        print(f"TESTING PAIR: {pair}")
        print(f"{'='*70}")
        
        pair_results = []
        
        for config in TIMEFRAME_CONFIGS:
            result = test_timeframe(pair, config['interval'], config['period'], config['name'])
            if result:
                pair_results.append(result)
                print(f"✅ {config['name']}: {result['trades']} trades, {result['win_rate']:.1f}% WR, {result['roi']:+.2f}% ROI, {result['trades_per_day']:.1f} trades/day")
                all_results.append(result)
        
        # Sort by ROI for this pair
        pair_results.sort(key=lambda x: x['roi'], reverse=True)
        
        print(f"\n🏆 BEST TIMEFRAMES FOR {pair}:")
        for i, r in enumerate(pair_results, 1):
            print(f"{i}. {r['config']}: {r['roi']:+.2f}% ROI, {r['win_rate']:.1f}% WR, {r['trades_per_day']:.1f} trades/day")
    
    # Overall results
    print(f"\n{'='*70}")
    print("OVERALL TIMEFRAME COMPARISON (Sorted by ROI)")
    print("="*70)
    print(f"{'Pair':<12} {'Timeframe':<28} {'Trades':<8} {'Win%':<7} {'ROI%':<10} {'Trades/Day':<12}")
    print("-"*70)
    
    all_results.sort(key=lambda x: x['roi'], reverse=True)
    for r in all_results:
        print(f"{r['pair']:<12} {r['config']:<28} {r['trades']:<8} {r['win_rate']:<7.1f} {r['roi']:<+10.2f} {r['trades_per_day']:<12.1f}")
    
    print("="*70)
    
    # Group by timeframe
    print(f"\n{'='*70}")
    print("PERFORMANCE BY TIMEFRAME (Average across pairs)")
    print("="*70)
    
    timeframe_stats = {}
    for result in all_results:
        config_name = result['config']
        if config_name not in timeframe_stats:
            timeframe_stats[config_name] = []
        timeframe_stats[config_name].append(result)
    
    timeframe_averages = []
    for config_name, results in timeframe_stats.items():
        avg_roi = sum(r['roi'] for r in results) / len(results)
        avg_wr = sum(r['win_rate'] for r in results) / len(results)
        avg_trades = sum(r['trades'] for r in results) / len(results)
        avg_tpd = sum(r['trades_per_day'] for r in results) / len(results)
        timeframe_averages.append({
            'config': config_name,
            'avg_roi': avg_roi,
            'avg_wr': avg_wr,
            'avg_trades': avg_trades,
            'avg_trades_per_day': avg_tpd
        })
    
    timeframe_averages.sort(key=lambda x: x['avg_roi'], reverse=True)
    
    print(f"{'Timeframe':<28} {'Avg ROI%':<12} {'Avg Win%':<10} {'Avg Trades/Day':<15}")
    print("-"*70)
    for t in timeframe_averages:
        print(f"{t['config']:<28} {t['avg_roi']:<+12.2f} {t['avg_wr']:<10.1f} {t['avg_trades_per_day']:<15.1f}")
    
    print("="*70)
    
    # Frequency analysis
    print(f"\n🎯 FREQUENCY ANALYSIS:")
    for t in timeframe_averages:
        if t['avg_trades_per_day'] >= 1.0:
            status = "✅ Meets target"
        else:
            status = "❌ Below target"
        print(f"  {t['config']:<28} {t['avg_trades_per_day']:>6.1f} trades/day {status}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scalping_timeframe_optimization_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'pairs_tested': BEST_PAIRS,
            'optimal_sl': OPTIMAL_SL,
            'optimal_tp': OPTIMAL_TP,
            'timeframes_tested': TIMEFRAME_CONFIGS,
            'results': all_results,
            'timeframe_averages': timeframe_averages
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    print("\n" + "="*70)
    print("TIMEFRAME OPTIMIZATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
