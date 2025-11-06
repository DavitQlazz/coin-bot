#!/usr/bin/env python3
"""
Run SMC backtest on multiple forex pairs and compare results
"""

import subprocess
import json
import os
from datetime import datetime

def run_backtest(symbol):
    """Run backtest for a single symbol"""
    print(f"\n{'='*60}")
    print(f"Testing {symbol}")
    print('='*60)
    
    try:
        result = subprocess.run(['python3', 'smc_backtest_4h.py', symbol], 
                              capture_output=True, text=True, cwd='/workspaces/coin-bot')
        
        # Print the output
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        return result.returncode == 0
    except Exception as e:
        print(f"Error running backtest for {symbol}: {e}")
        return False

def load_results(symbol):
    """Load results for a symbol"""
    safe_symbol = symbol.replace('/', '_').replace('=X', '')
    fname = f'smc_backtest_results_{safe_symbol}.json'
    
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            return json.load(f)
    return None

def main():
    # Forex pairs to test
    forex_pairs = [
        'EURUSD=X',
        'GBPUSD=X', 
        'USDJPY=X',
        'AUDUSD=X',
        'USDCAD=X',
        'USDCHF=X'
    ]
    
    results = []
    
    # Run backtests
    for pair in forex_pairs:
        if run_backtest(pair):
            result_data = load_results(pair)
            if result_data:
                summary = result_data['summary']
                results.append({
                    'pair': pair,
                    'trades': summary['trades'],
                    'wins': summary['wins'],
                    'losses': summary['losses'],
                    'win_rate': summary['wins'] / summary['trades'] * 100 if summary['trades'] > 0 else 0,
                    'net_pnl': summary['net_pnl'],
                    'ending_balance': summary['ending_balance']
                })
    
    # Print comparison
    print(f"\n{'='*80}")
    print("SMC BACKTEST COMPARISON - FOREX PAIRS (4h, 180 days)")
    print('='*80)
    print(f"{'Pair':<10} {'Trades':<6} {'Wins':<5} {'Losses':<7} {'Win%':<6} {'Net P/L':<10} {'End Balance':<12}")
    print('-'*80)
    
    for result in results:
        print(f"{result['pair']:<10} {result['trades']:<6} {result['wins']:<5} {result['losses']:<7} "
              f"{result['win_rate']:<6.1f} ${result['net_pnl']:<9.0f} ${result['ending_balance']:<11.0f}")
    
    # Summary stats
    if results:
        avg_win_rate = sum(r['win_rate'] for r in results) / len(results)
        total_trades = sum(r['trades'] for r in results)
        total_pnl = sum(r['net_pnl'] for r in results)
        
        print('-'*80)
        print(f"{'Average':<10} {total_trades/len(results):<6.1f} {'-':<5} {'-':<7} "
              f"{avg_win_rate:<6.1f} ${total_pnl:<9.0f} {'-':<12}")
        
        # Best performer
        best = max(results, key=lambda x: x['net_pnl'])
        print(f"\nBest performer: {best['pair']} with ${best['net_pnl']:.0f} P/L")
        
        # Worst performer  
        worst = min(results, key=lambda x: x['net_pnl'])
        print(f"Worst performer: {worst['pair']} with ${worst['net_pnl']:.0f} P/L")

if __name__ == '__main__':
    main()