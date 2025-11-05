"""
Forex Backtesting Module
Backtest forex trading strategies on multiple currency pairs using Yahoo Finance data
"""

from bot_forex import ForexTradingBot
import pandas as pd
import json
from datetime import datetime

class ForexBacktester(ForexTradingBot):
    """Extended ForexTradingBot for comprehensive backtesting"""
    
    def __init__(self, pair='EURUSD=X'):
        super().__init__()
        self.symbol = pair
        print(f"🔬 Backtester initialized for {pair}")
    
    def backtest(self, period='30d', interval='1h'):
        """
        Run backtest on specific forex pair
        
        Args:
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', 'max'
            interval: '1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'
        """
        self.interval = interval
        return self.run(period=period)
    
    def backtest_multiple_pairs(self, pairs, period='30d', interval='1h'):
        """
        Backtest multiple forex pairs and compare results
        
        Args:
            pairs: List of forex pair symbols (Yahoo Finance format)
            period: Historical data period
            interval: Timeframe for candles
        """
        print(f"\n{'='*80}")
        print(f"{'MULTI-PAIR FOREX BACKTESTING':^80}")
        print(f"{'='*80}")
        print(f"Testing {len(pairs)} pairs with {period} of {interval} data")
        print(f"Parameters: Stop Loss={self.stop_loss_pct}%, Take Profit={self.take_profit_pct}%")
        print(f"{'='*80}\n")
        
        results = []
        
        for pair in pairs:
            print(f"\n{'─'*80}")
            print(f"🔍 Testing: {pair}")
            print(f"{'─'*80}")
            
            # Reset bot state for each pair
            self.symbol = pair
            self.capital = self.initial_capital
            self.position = None
            self.trades = []
            
            # Run backtest
            try:
                metrics = self.run(period=period)
                
                if metrics:
                    results.append({
                        'pair': pair,
                        'interval': interval,
                        'period': period,
                        **metrics
                    })
                else:
                    print(f"⚠️  No results for {pair}")
                    
            except Exception as e:
                print(f"❌ Error testing {pair}: {e}")
                continue
        
        # Print comparison
        if results:
            self._print_comparison(results)
            self._save_results(results)
        
        return results
    
    def _print_comparison(self, results):
        """Print comparison table of all backtesting results"""
        print(f"\n{'='*80}")
        print(f"{'BACKTESTING RESULTS COMPARISON':^80}")
        print(f"{'='*80}")
        
        df = pd.DataFrame(results)
        
        # Sort by ROI
        df = df.sort_values('roi', ascending=False)
        
        print(f"\n{'Pair':<15} {'Trades':<8} {'Win Rate':<10} {'ROI':<10} {'P.Factor':<10} {'Final $':<12}")
        print(f"{'-'*80}")
        
        for _, row in df.iterrows():
            print(f"{row['pair']:<15} {row['total_trades']:<8} {row['win_rate']:>6.2f}%   {row['roi']:>6.2f}%   {row['profit_factor']:>6.2f}    ${row['final_capital']:>10,.2f}")
        
        print(f"\n{'='*80}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Best ROI: {df.iloc[0]['pair']} ({df.iloc[0]['roi']:.2f}%)")
        print(f"Highest Win Rate: {df.loc[df['win_rate'].idxmax()]['pair']} ({df['win_rate'].max():.2f}%)")
        print(f"Most Trades: {df.loc[df['total_trades'].idxmax()]['pair']} ({df['total_trades'].max()} trades)")
        print(f"Average ROI: {df['roi'].mean():.2f}%")
        print(f"Average Win Rate: {df['win_rate'].mean():.2f}%")
        print(f"Profitable Pairs: {len(df[df['roi'] > 0])}/{len(df)}")
        print(f"{'='*80}\n")
    
    def _save_results(self, results):
        """Save backtesting results to JSON file"""
        filename = f"forex_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}\n")


def run_comprehensive_backtest():
    """Run comprehensive backtest on major forex pairs"""
    
    # Major forex pairs (Yahoo Finance format)
    major_pairs = [
        'EURUSD=X',  # Euro / US Dollar
        'GBPUSD=X',  # British Pound / US Dollar
        'USDJPY=X',  # US Dollar / Japanese Yen
        'AUDUSD=X',  # Australian Dollar / US Dollar
    ]
    
    print(f"\n{'='*80}")
    print(f"{'COMPREHENSIVE FOREX BACKTESTING':^80}")
    print(f"{'='*80}")
    print(f"Testing {len(major_pairs)} major currency pairs")
    print(f"Strategy: Multi-indicator trend following")
    print(f"Data source: Yahoo Finance")
    print(f"{'='*80}\n")
    
    # Initialize backtester
    backtester = ForexBacktester()
    
    # Test with 30 days of 1-hour data
    results = backtester.backtest_multiple_pairs(
        pairs=major_pairs,
        period='30d',
        interval='1h'
    )
    
    return results


def test_single_pair(pair='EURUSD=X', period='30d', interval='1h'):
    """Test a single forex pair with detailed output"""
    print(f"\n{'='*80}")
    print(f"SINGLE PAIR BACKTEST: {pair}")
    print(f"{'='*80}\n")
    
    backtester = ForexBacktester(pair=pair)
    metrics = backtester.backtest(period=period, interval=interval)
    
    return metrics


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Test single pair if provided
        pair = sys.argv[1]
        test_single_pair(pair)
    else:
        # Run comprehensive backtest on all major pairs
        run_comprehensive_backtest()
