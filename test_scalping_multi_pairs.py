#!/usr/bin/env python3
"""
Test scalping strategy on multiple forex pairs to find best performers
"""

import subprocess
import json
import os
from datetime import datetime

# Test configuration
PAIRS_TO_TEST = [
    'EURUSD=X',
    'GBPUSD=X',
    'USDJPY=X',
    'USDCHF=X',
    'AUDUSD=X',
    'NZDUSD=X',
    'USDCAD=X',
    'EURJPY=X',
    'EURGBP=X',
    'GBPJPY=X'
]

TEST_PERIOD = 7  # days (Yahoo Finance 5m data limit)

def test_pair(pair):
    """Test scalping strategy on a single pair"""
    print(f"\n{'='*70}")
    print(f"Testing: {pair}")
    print(f"{'='*70}")
    
    # Modify bot_forex_scalping.py temporarily to test this pair
    # Read the file
    with open('bot_forex_scalping.py', 'r') as f:
        content = f.read()
    
    # Backup original
    original_pair_line = None
    for line in content.split('\n'):
        if 'self.symbol = ' in line and 'EURUSD=X' in line:
            original_pair_line = line
            break
    
    if not original_pair_line:
        print(f"⚠️  Could not find symbol line in bot_forex_scalping.py")
        return None
    
    # Replace pair
    new_content = content.replace(original_pair_line, f"        self.symbol = '{pair}'  # Testing {pair}")
    
    with open('bot_forex_scalping.py', 'w') as f:
        f.write(new_content)
    
    # Run the test
    try:
        result = subprocess.run(['python3', 'bot_forex_scalping.py'], 
                              capture_output=True, 
                              text=True, 
                              timeout=60)
        output = result.stdout
        
        # Restore original
        with open('bot_forex_scalping.py', 'w') as f:
            f.write(content)
        
        # Parse results from output
        results = parse_output(output, pair)
        return results
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout testing {pair}")
        # Restore original
        with open('bot_forex_scalping.py', 'w') as f:
            f.write(content)
        return None
    except Exception as e:
        print(f"❌ Error testing {pair}: {e}")
        # Restore original
        with open('bot_forex_scalping.py', 'w') as f:
            f.write(content)
        return None

def parse_output(output, pair):
    """Parse backtesting output to extract metrics"""
    results = {
        'pair': pair,
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0.0,
        'roi': 0.0,
        'profit_factor': 0.0,
        'avg_hold_time': 0.0,
        'trades_per_day': 0.0,
        'final_capital': 1000.0
    }
    
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if 'Total Trades:' in line:
            results['trades'] = int(line.split(':')[1].strip())
        elif 'Wins:' in line and '(' in line:
            parts = line.split(':')[1].strip().split('(')
            results['wins'] = int(parts[0].strip())
            results['win_rate'] = float(parts[1].replace('%)', '').strip())
        elif 'Losses:' in line:
            results['losses'] = int(line.split(':')[1].strip())
        elif 'Total Return:' in line and '(' in line:
            parts = line.split('(')[1].split('%')[0]
            results['roi'] = float(parts)
        elif 'Profit Factor:' in line:
            pf = line.split(':')[1].strip()
            if pf != 'inf':
                results['profit_factor'] = float(pf)
            else:
                results['profit_factor'] = 999.0
        elif 'Avg Hold Time:' in line:
            results['avg_hold_time'] = float(line.split(':')[1].strip().split()[0])
        elif 'Trades/Day:' in line:
            results['trades_per_day'] = float(line.split(':')[1].strip())
        elif 'Final Capital:' in line:
            capital = line.split('$')[1].strip()
            results['final_capital'] = float(capital)
    
    return results

def main():
    print("="*70)
    print("SCALPING STRATEGY - MULTI-PAIR COMPARISON TEST")
    print("="*70)
    print(f"Testing {len(PAIRS_TO_TEST)} forex pairs")
    print(f"Period: {TEST_PERIOD} days")
    print(f"Interval: 5 minutes")
    print(f"Stop Loss: 0.4% | Take Profit: 0.8%")
    print("="*70)
    
    all_results = []
    
    for pair in PAIRS_TO_TEST:
        results = test_pair(pair)
        if results:
            all_results.append(results)
            print(f"\n✅ {pair}: {results['trades']} trades, {results['win_rate']:.1f}% WR, {results['roi']:+.2f}% ROI, {results['trades_per_day']:.1f} trades/day")
        else:
            print(f"\n❌ {pair}: Failed to test")
    
    # Sort by ROI
    all_results.sort(key=lambda x: x['roi'], reverse=True)
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY - SORTED BY ROI")
    print("="*70)
    print(f"{'Pair':<12} {'Trades':<8} {'Win%':<8} {'ROI%':<10} {'Trades/Day':<12} {'PF':<8}")
    print("-"*70)
    
    for r in all_results:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 900 else "∞"
        print(f"{r['pair']:<12} {r['trades']:<8} {r['win_rate']:<8.1f} {r['roi']:<+10.2f} {r['trades_per_day']:<12.1f} {pf_str:<8}")
    
    # Calculate averages
    if all_results:
        avg_roi = sum(r['roi'] for r in all_results) / len(all_results)
        avg_wr = sum(r['win_rate'] for r in all_results) / len(all_results)
        avg_trades = sum(r['trades'] for r in all_results) / len(all_results)
        avg_tpd = sum(r['trades_per_day'] for r in all_results) / len(all_results)
        
        print("-"*70)
        print(f"{'AVERAGE':<12} {avg_trades:<8.0f} {avg_wr:<8.1f} {avg_roi:<+10.2f} {avg_tpd:<12.1f}")
        print("="*70)
        
        # Best performers
        print("\n🏆 TOP 3 PERFORMERS BY ROI:")
        for i, r in enumerate(all_results[:3], 1):
            print(f"{i}. {r['pair']}: {r['roi']:+.2f}% ROI, {r['win_rate']:.1f}% WR, {r['trades']} trades")
        
        # Most active
        most_active = sorted(all_results, key=lambda x: x['trades_per_day'], reverse=True)
        print("\n📊 TOP 3 MOST ACTIVE (Trades/Day):")
        for i, r in enumerate(most_active[:3], 1):
            print(f"{i}. {r['pair']}: {r['trades_per_day']:.1f} trades/day, {r['roi']:+.2f}% ROI")
        
        # Best win rate
        best_wr = sorted(all_results, key=lambda x: x['win_rate'], reverse=True)
        print("\n🎯 TOP 3 BEST WIN RATES:")
        for i, r in enumerate(best_wr[:3], 1):
            print(f"{i}. {r['pair']}: {r['win_rate']:.1f}% WR, {r['roi']:+.2f}% ROI")
        
        # Frequency target check
        print("\n🎯 FREQUENCY TARGET (1+ trade/day):")
        meeting_target = [r for r in all_results if r['trades_per_day'] >= 1.0]
        print(f"✅ {len(meeting_target)}/{len(all_results)} pairs meet target")
        for r in meeting_target:
            print(f"   {r['pair']}: {r['trades_per_day']:.1f} trades/day")
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scalping_multi_pair_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'test_period_days': TEST_PERIOD,
                'pairs_tested': len(all_results),
                'results': all_results,
                'summary': {
                    'avg_roi': avg_roi,
                    'avg_win_rate': avg_wr,
                    'avg_trades': avg_trades,
                    'avg_trades_per_day': avg_tpd,
                    'pairs_meeting_frequency_target': len(meeting_target)
                }
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    print("\n" + "="*70)
    print("MULTI-PAIR TEST COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
