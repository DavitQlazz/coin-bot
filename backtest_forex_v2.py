"""
Enhanced Forex Backtester V2 with improved strategy
"""
from bot_forex_v2 import ForexTradingBotV2
from datetime import datetime
import pandas as pd

class ForexBacktesterV2(ForexTradingBotV2):
    """Backtester for enhanced forex strategy"""
    
    def __init__(self, pair='EURUSD=X', stop_loss=2.6, take_profit=6.2, min_adx=25, min_signal_score=3):
        super().__init__()
        self.symbol = pair
        self.stop_loss_pct = stop_loss
        self.take_profit_pct = take_profit
        self.min_adx = min_adx
        self.min_signal_score = min_signal_score
        
    def backtest(self, period='360d', interval='4h', verbose=False):
        """Run backtest on historical data"""
        self.interval = interval
        
        print(f"\n{'='*70}")
        print(f"BACKTESTING V2: {self.symbol} | Period: {period} | Interval: {interval}")
        print(f"Stop Loss: {self.stop_loss_pct}% | Take Profit: {self.take_profit_pct}%")
        print(f"Min ADX: {self.min_adx} | Min Signal Score: {self.min_signal_score}")
        print(f"{'='*70}\n")
        
        # Fetch data
        df = self.fetch_ohlcv(period)
        if df is None:
            return None
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Simulate trading
        for i in range(200, len(df)):  # Start after 200 periods for EMA200
            current_data = df.iloc[:i+1].copy()
            current_price = df.iloc[i]['close']
            timestamp = df.iloc[i]['timestamp']
            
            # Check exit conditions first
            if self.position is not None:
                self.check_exit_conditions(current_price, timestamp)
            
            # Generate signal
            if self.position is None:
                signal, confidence = self.generate_signal(current_data)
                
                if verbose and signal != 'HOLD':
                    trend, adx = self.calculate_trend_strength(current_data)
                    print(f"[{timestamp}] Signal: {signal} | Confidence: {confidence}/7 | ADX: {adx:.1f} | Trend: {trend}")
                
                if signal in ['BUY', 'SELL']:
                    self.execute_trade(signal, confidence, current_price, timestamp)
        
        # Close any open position at end
        if self.position is not None:
            self.close_position(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], "End of Test")
        
        # Calculate metrics
        metrics = self.get_performance_metrics()
        
        if metrics:
            print(f"\n{'='*70}")
            print(f"BACKTEST RESULTS V2: {self.symbol}")
            print(f"{'='*70}")
            print(f"Total Trades:     {metrics['total_trades']}")
            print(f"Wins:             {metrics['wins']} ({metrics['win_rate']:.1f}%)")
            print(f"Losses:           {metrics['losses']}")
            print(f"Total Return:     ${metrics['total_return']:.2f} ({metrics['return_pct']:.2f}%)")
            print(f"Avg Win:          ${metrics['avg_win']:.2f}")
            print(f"Avg Loss:         ${metrics['avg_loss']:.2f}")
            print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
            print(f"Final Capital:    ${metrics['final_capital']:,.2f}")
            print(f"{'='*70}\n")
        
        return metrics


def compare_strategies(pair, period='360d', interval='4h'):
    """Compare V1 vs V2 strategy"""
    from backtest_forex import ForexBacktester
    
    print(f"\n{'#'*70}")
    print(f"STRATEGY COMPARISON: {pair}")
    print(f"Period: {period} | Interval: {interval}")
    print(f"{'#'*70}\n")
    
    # Test V1 (Original Strategy)
    print("\n--- STRATEGY V1 (Original) ---")
    bot_v1 = ForexBacktester(pair=pair)
    bot_v1.stop_loss_pct = 2.6
    bot_v1.take_profit_pct = 6.2
    metrics_v1 = bot_v1.backtest(period, interval)
    
    # Test V2 (Enhanced Strategy)
    print("\n--- STRATEGY V2 (Enhanced with ADX) ---")
    bot_v2 = ForexBacktesterV2(
        pair=pair,
        stop_loss=2.6,
        take_profit=6.2,
        min_adx=25,
        min_signal_score=3
    )
    metrics_v2 = bot_v2.backtest(period, interval)
    
    # Comparison
    if metrics_v1 and metrics_v2:
        # Handle different key names (roi vs return_pct)
        roi_v1 = metrics_v1.get('return_pct', metrics_v1.get('roi', 0))
        roi_v2 = metrics_v2.get('return_pct', metrics_v2.get('roi', 0))
        
        print(f"\n{'='*70}")
        print(f"COMPARISON: V1 vs V2")
        print(f"{'='*70}")
        print(f"{'Metric':<20} {'V1 (Original)':<20} {'V2 (Enhanced)':<20} {'Change':<15}")
        print(f"{'-'*70}")
        print(f"{'ROI':<20} {roi_v1:>19.2f}% {roi_v2:>19.2f}% {roi_v2-roi_v1:>+14.2f}%")
        print(f"{'Trades':<20} {metrics_v1['total_trades']:>19} {metrics_v2['total_trades']:>19} {metrics_v2['total_trades']-metrics_v1['total_trades']:>+14}")
        print(f"{'Win Rate':<20} {metrics_v1['win_rate']:>18.1f}% {metrics_v2['win_rate']:>18.1f}% {metrics_v2['win_rate']-metrics_v1['win_rate']:>+13.1f}%")
        print(f"{'Profit Factor':<20} {metrics_v1['profit_factor']:>19.2f} {metrics_v2['profit_factor']:>19.2f} {metrics_v2['profit_factor']-metrics_v1['profit_factor']:>+14.2f}")
        print(f"{'Avg Win':<20} ${metrics_v1['avg_win']:>18.2f} ${metrics_v2['avg_win']:>18.2f} ${metrics_v2['avg_win']-metrics_v1['avg_win']:>+13.2f}")
        print(f"{'Avg Loss':<20} ${metrics_v1['avg_loss']:>18.2f} ${metrics_v2['avg_loss']:>18.2f} ${metrics_v2['avg_loss']-metrics_v1['avg_loss']:>+13.2f}")
        print(f"{'='*70}\n")
        
        # Verdict
        if roi_v2 > roi_v1:
            improvement = roi_v2 - roi_v1
            print(f"✅ V2 WINS: {improvement:+.2f}% better ROI!")
        elif roi_v2 < roi_v1:
            decline = roi_v1 - roi_v2
            print(f"❌ V1 WINS: {decline:.2f}% better ROI")
        else:
            print(f"🤝 TIE: Same performance")
    
    return metrics_v1, metrics_v2


def test_all_pairs_v2(period='360d', interval='4h'):
    """Test V2 strategy on all major forex pairs"""
    pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
    
    print(f"\n{'#'*70}")
    print(f"TESTING V2 STRATEGY ON ALL PAIRS")
    print(f"Period: {period} | Interval: {interval}")
    print(f"{'#'*70}\n")
    
    results = []
    
    for pair in pairs:
        bot = ForexBacktesterV2(
            pair=pair,
            stop_loss=2.6,
            take_profit=6.2,
            min_adx=25,
            min_signal_score=3
        )
        
        metrics = bot.backtest(period, interval)
        if metrics:
            results.append({
                'pair': pair,
                **metrics
            })
    
    # Summary
    if results:
        print(f"\n{'='*90}")
        print(f"V2 STRATEGY SUMMARY - ALL PAIRS")
        print(f"{'='*90}")
        print(f"{'Pair':<12} {'ROI':<10} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<15} {'Avg Win':<12} {'Avg Loss':<12}")
        print(f"{'-'*90}")
        
        for r in results:
            print(f"{r['pair']:<12} {r['return_pct']:>8.2f}% {r['total_trades']:>7} {r['win_rate']:>8.1f}% {r['profit_factor']:>14.2f} ${r['avg_win']:>10.2f} ${r['avg_loss']:>10.2f}")
        
        avg_roi = sum(r['return_pct'] for r in results) / len(results)
        avg_wr = sum(r['win_rate'] for r in results) / len(results)
        profitable = sum(1 for r in results if r['return_pct'] > 0)
        
        print(f"{'-'*90}")
        print(f"{'AVERAGE':<12} {avg_roi:>8.2f}% {'-':>7} {avg_wr:>8.1f}% {'-':>14} {'-':>11} {'-':>11}")
        print(f"{'Profitable:':<12} {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)")
        print(f"{'='*90}\n")
    
    return results


if __name__ == '__main__':
    # Test single pair with comparison
    print("\n🚀 Testing Enhanced Strategy V2...\n")
    
    # Test on ALL pairs to see overall improvement
    print("\n" + "="*70)
    print("TESTING V2 ON ALL PAIRS")
    print("="*70 + "\n")
    
    pairs_to_test = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
    
    for pair in pairs_to_test:
        print(f"\n{'#'*70}")
        print(f"PAIR: {pair}")
        print(f"{'#'*70}")
        compare_strategies(pair, period='360d', interval='4h')
    
    # Test all pairs with V2
    # test_all_pairs_v2(period='360d', interval='4h')
