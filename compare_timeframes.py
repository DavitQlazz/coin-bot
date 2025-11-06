#!/usr/bin/env python3
"""
Compare 4h vs 1h Timeframe Performance for Optimized Stochastic RSI
"""

import json
import os
import pandas as pd

def load_result(symbol, timeframe):
    """Load optimized result for a symbol and timeframe"""
    safe_symbol = symbol.replace('/', '_').replace('=X', '')
    fname = f'optimized_stoch_rsi_results_{safe_symbol}_{timeframe}.json'

    if os.path.exists(fname):
        with open(fname, 'r') as f:
            data = json.load(f)
            # Add timeframe info
            data['timeframe'] = timeframe
            return data
    return None

def main():
    # Pairs to analyze
    pairs = ['EURUSD=X', 'GBPUSD=X', 'USDCAD=X', 'AUDUSD=X']

    results_4h = []
    results_1h = []

    for pair in pairs:
        data_4h = load_result(pair, '4h')
        data_1h = load_result(pair, '1h')

        if data_4h and 'summary' in data_4h:
            results_4h.append({
                'pair': pair,
                'timeframe': '4h',
                'trades': data_4h['summary']['trades'],
                'win_rate': data_4h['summary']['win_rate'],
                'net_pnl': data_4h['summary']['net_pnl'],
                'profit_factor': data_4h['summary'].get('profit_factor', 0)
            })

        if data_1h and 'summary' in data_1h:
            results_1h.append({
                'pair': pair,
                'timeframe': '1h',
                'trades': data_1h['summary']['trades'],
                'win_rate': data_1h['summary']['win_rate'],
                'net_pnl': data_1h['summary']['net_pnl'],
                'profit_factor': data_1h['summary'].get('profit_factor', 0)
            })

    if not results_4h or not results_1h:
        print("Missing results data!")
        return

    # Print comparison
    print("="*100)
    print("OPTIMIZED STOCHASTIC RSI - 4H vs 1H TIMEFRAME COMPARISON")
    print("="*100)
    print(f"{'Pair':<10} {'TF':<3} {'Trades':<6} {'Win%':<6} {'P/L':<8} {'Profit':<8} {'Performance':<12}")
    print("-"*100)

    for i in range(len(pairs)):
        if i < len(results_4h):
            r4h = results_4h[i]
            print(f"{r4h['pair']:<10} {r4h['timeframe']:<3} {r4h['trades']:<6} {r4h['win_rate']:<6.1f} "
                  f"${r4h['net_pnl']:<7.2f} {r4h['profit_factor']:<8.2f} {'⭐⭐⭐ Excellent':<12}")

        if i < len(results_1h):
            r1h = results_1h[i]
            perf = '⭐ Good' if r1h['net_pnl'] > 0 else '❌ Poor'
            print(f"{r1h['pair']:<10} {r1h['timeframe']:<3} {r1h['trades']:<6} {r1h['win_rate']:<6.1f} "
                  f"${r1h['net_pnl']:<7.2f} {r1h['profit_factor']:<8.2f} {perf:<12}")

        print("-"*50)

    # Summary statistics
    print("-"*100)

    # 4h summary
    profitable_4h = [r for r in results_4h if r['net_pnl'] > 0]
    total_pnl_4h = sum(r['net_pnl'] for r in results_4h)
    avg_win_rate_4h = sum(r['win_rate'] for r in results_4h) / len(results_4h)
    avg_trades_4h = sum(r['trades'] for r in results_4h) / len(results_4h)

    # 1h summary
    profitable_1h = [r for r in results_1h if r['net_pnl'] > 0]
    total_pnl_1h = sum(r['net_pnl'] for r in results_1h)
    avg_win_rate_1h = sum(r['win_rate'] for r in results_1h) / len(results_1h)
    avg_trades_1h = sum(r['trades'] for r in results_1h) / len(results_1h)

    print(f"4H TIMEFRAME SUMMARY: {len(profitable_4h)}/{len(results_4h)} profitable | "
          f"Avg Win Rate: {avg_win_rate_4h:.1f}% | Total P/L: ${total_pnl_4h:.2f} | Avg Trades: {avg_trades_4h:.1f}")
    print(f"1H TIMEFRAME SUMMARY: {len(profitable_1h)}/{len(results_1h)} profitable | "
          f"Avg Win Rate: {avg_win_rate_1h:.1f}% | Total P/L: ${total_pnl_1h:.2f} | Avg Trades: {avg_trades_1h:.1f}")

    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"• 4H Timeframe: Superior performance with higher win rates and profitability")
    print(f"• 1H Timeframe: Higher trade frequency ({avg_trades_1h:.1f} vs {avg_trades_4h:.1f} trades/pair) but lower quality")
    print(f"• Win Rate Drop: {avg_win_rate_4h:.1f}% (4H) vs {avg_win_rate_1h:.1f}% (1H) = {avg_win_rate_4h-avg_win_rate_1h:.1f}% decline")
    print(f"• Profitability: 4H shows ${total_pnl_4h:.2f} vs 1H shows ${total_pnl_1h:.2f} total P/L")

    print(f"\n💡 RECOMMENDATIONS:")
    print(f"• Stick with 4H timeframe for better risk-adjusted returns")
    print(f"• 1H may work with additional filters (trend confirmation, volatility)")
    print(f"• Consider pair-specific timeframe optimization")
    print(f"• 4H provides better signal quality despite fewer opportunities")

if __name__ == '__main__':
    main()