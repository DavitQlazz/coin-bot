#!/usr/bin/env python3
"""
Optimize scalping SL/TP ratios on best performing pairs
"""

from bot_forex_scalping import ForexScalpingBot
from datetime import datetime
import json

# Best performing pairs from initial test
BEST_PAIRS = ['GBPUSD=X', 'USDJPY=X']

# SL/TP combinations to test
SL_TP_CONFIGS = [
    {'sl': 0.3, 'tp': 0.6, 'name': '1:2 ratio (tight)'},
    {'sl': 0.4, 'tp': 0.8, 'name': '1:2 ratio (current)'},
    {'sl': 0.5, 'tp': 1.0, 'name': '1:2 ratio (loose)'},
    {'sl': 0.4, 'tp': 1.2, 'name': '1:3 ratio'},
    {'sl': 0.3, 'tp': 0.9, 'name': '1:3 ratio (tight)'},
    {'sl': 0.5, 'tp': 1.5, 'name': '1:3 ratio (loose)'},
    {'sl': 0.6, 'tp': 0.9, 'name': '1:1.5 ratio (conservative)'},
]

def test_config(pair, sl, tp, config_name):
    """Test a single SL/TP configuration"""
    try:
        bot = ForexScalpingBot(
            pair=pair,
            stop_loss=sl,
            take_profit=tp,
            max_hold_minutes=30
        )
        
        results = bot.backtest(period='7d', interval='5m')
        
        return {
            'pair': pair,
            'config': config_name,
            'sl': sl,
            'tp': tp,
            'trades': results['total_trades'],
            'wins': results['wins'],
            'win_rate': results['win_rate'],
            'roi': results['roi'],
            'profit_factor': results['profit_factor'],
            'trades_per_day': results['trades_per_day']
        }
    except Exception as e:
        print(f"❌ Error testing {pair} with SL={sl}% TP={tp}%: {e}")
        return None

def main():
    print("="*70)
    print("SCALPING STRATEGY - SL/TP OPTIMIZATION")
    print("="*70)
    print(f"Testing {len(SL_TP_CONFIGS)} configurations on {len(BEST_PAIRS)} pairs")
    print(f"Period: 7 days | Interval: 5 minutes")
    print("="*70)
    
    all_results = []
    
    for pair in BEST_PAIRS:
        print(f"\n{'='*70}")
        print(f"TESTING PAIR: {pair}")
        print(f"{'='*70}\n")
        
        pair_results = []
        
        for config in SL_TP_CONFIGS:
            print(f"Testing {config['name']}: SL={config['sl']}% TP={config['tp']}%")
            result = test_config(pair, config['sl'], config['tp'], config['name'])
            if result:
                pair_results.append(result)
                print(f"✅ {result['trades']} trades, {result['win_rate']:.1f}% WR, {result['roi']:+.2f}% ROI\n")
                all_results.append(result)
        
        # Sort by ROI for this pair
        pair_results.sort(key=lambda x: x['roi'], reverse=True)
        
        print(f"\n🏆 BEST CONFIGS FOR {pair}:")
        for i, r in enumerate(pair_results[:3], 1):
            ratio = r['tp'] / r['sl']
            print(f"{i}. {r['config']}: {r['roi']:+.2f}% ROI, {r['win_rate']:.1f}% WR, {r['trades']} trades")
    
    # Overall best configurations
    print(f"\n{'='*70}")
    print("OVERALL BEST CONFIGURATIONS (Sorted by ROI)")
    print("="*70)
    print(f"{'Pair':<12} {'Config':<25} {'SL%':<6} {'TP%':<6} {'Win%':<7} {'ROI%':<10} {'Trades':<8}")
    print("-"*70)
    
    all_results.sort(key=lambda x: x['roi'], reverse=True)
    for r in all_results[:10]:
        print(f"{r['pair']:<12} {r['config']:<25} {r['sl']:<6} {r['tp']:<6} {r['win_rate']:<7.1f} {r['roi']:<+10.2f} {r['trades']:<8}")
    
    print("="*70)
    
    # Group by configuration
    print(f"\n{'='*70}")
    print("PERFORMANCE BY CONFIGURATION (Average across pairs)")
    print("="*70)
    
    config_stats = {}
    for result in all_results:
        config_name = result['config']
        if config_name not in config_stats:
            config_stats[config_name] = []
        config_stats[config_name].append(result)
    
    config_averages = []
    for config_name, results in config_stats.items():
        avg_roi = sum(r['roi'] for r in results) / len(results)
        avg_wr = sum(r['win_rate'] for r in results) / len(results)
        avg_trades = sum(r['trades'] for r in results) / len(results)
        config_averages.append({
            'config': config_name,
            'avg_roi': avg_roi,
            'avg_wr': avg_wr,
            'avg_trades': avg_trades
        })
    
    config_averages.sort(key=lambda x: x['avg_roi'], reverse=True)
    
    print(f"{'Config':<25} {'Avg ROI%':<12} {'Avg Win%':<10} {'Avg Trades':<12}")
    print("-"*70)
    for c in config_averages:
        print(f"{c['config']:<25} {c['avg_roi']:<+12.2f} {c['avg_wr']:<10.1f} {c['avg_trades']:<12.0f}")
    
    print("="*70)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scalping_sltp_optimization_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'pairs_tested': BEST_PAIRS,
            'configs_tested': SL_TP_CONFIGS,
            'results': all_results,
            'config_averages': config_averages
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    print("\n" + "="*70)
    print("SL/TP OPTIMIZATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
