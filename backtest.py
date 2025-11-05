import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from bot import CryptoTradingBot
import ccxt

class Backtester(CryptoTradingBot):
    def __init__(self):
        # Initialize without API keys for public data access
        self.symbol = 'BTC/USDT'
        self.timeframe = '1h'
        self.capital = 1000.0
        self.initial_capital = self.capital
        self.position = None
        self.trades = []
        self.paper_trading = True  # Always paper trade in backtest
        self.stop_loss_pct = 2.0
        self.take_profit_pct = 5.0
        self.equity_curve = []
        
        # Initialize Binance exchange for public data (no API keys needed)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        try:
            self.exchange.load_markets()
            print("✅ Connected to Binance (public API - no authentication required)")
        except Exception as e:
            print(f"⚠️  Connection warning: {e}")
            print("Continuing with available functionality...")
        
    def backtest(self, days=30, symbol=None, timeframe=None):
        """Run backtest on historical data using Binance public API"""
        # Allow override of default settings
        if symbol:
            self.symbol = symbol
        if timeframe:
            self.timeframe = timeframe
            
        print(f"\n{'='*60}")
        print(f"🔄 BACKTESTING MODE - Binance Public API")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Period: {days} days")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Data Source: Binance (Public API)")
        print(f"{'='*60}\n")
        
        # Calculate required candles
        if self.timeframe == '1h':
            limit = days * 24
        elif self.timeframe == '4h':
            limit = days * 6
        elif self.timeframe == '1d':
            limit = days
        elif self.timeframe == '15m':
            limit = days * 96
        elif self.timeframe == '5m':
            limit = days * 288
        else:
            limit = days * 24  # Default to hourly
        
        # Fetch historical data from Binance public API
        print(f"📥 Fetching {min(limit, 1000)} candles from Binance...")
        df = self.fetch_ohlcv(limit=min(limit, 1000))  # Binance limit
        
        if df is None or len(df) == 0:
            print("❌ Failed to fetch data. Check your API connection.")
            return
        
        print(f"✅ Received {len(df)} candles")
        print(f"📅 Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}\n")
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Run backtest
        print("🔄 Running backtest...\n")
        
        for i in range(50, len(df)):  # Start after indicator warmup period
            window = df.iloc[:i+1]
            signal = self.generate_signal(window)
            current_price = window.iloc[-1]['close']
            timestamp = window.iloc[-1]['timestamp']
            
            # Execute trade
            old_capital = self.capital
            self.execute_trade(signal, current_price)
            
            # Track equity
            current_equity = self.capital
            if self.position:
                unrealized_pnl = (current_price - self.position['entry_price']) * self.position['amount']
                current_equity += unrealized_pnl
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'price': current_price
            })
        
        # Close any open position at the end
        if self.position:
            print("\n⚠️  Closing open position at backtest end...")
            final_price = df.iloc[-1]['close']
            self.execute_trade('SELL', final_price)
        
        # Print results
        self.print_backtest_results()
        self.plot_results(df)
    
    def print_backtest_results(self):
        """Print detailed backtest results"""
        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS")
        print("="*60)
        
        if not self.trades:
            print("❌ No trades executed during backtest period.")
            print("This could mean:")
            print("  - Market conditions didn't meet entry criteria")
            print("  - Strategy parameters are too strict")
            print("  - Insufficient data or volatility")
            print("="*60)
            return
        
        # Basic statistics
        total_profit = sum(t['profit'] for t in self.trades)
        winning_trades = [t for t in self.trades if t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['profit'] < 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100
        roi = ((self.capital / self.initial_capital) - 1) * 100
        
        print(f"\n📈 PERFORMANCE METRICS")
        print(f"-" * 60)
        print(f"Total Trades: {len(self.trades)}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)} ({100-win_rate:.1f}%)")
        
        print(f"\n💰 PROFIT & LOSS")
        print(f"-" * 60)
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Net Profit/Loss: ${total_profit:+.2f}")
        print(f"ROI: {roi:+.2f}%")
        
        if winning_trades:
            total_wins = sum(t['profit'] for t in winning_trades)
            avg_win = total_wins / len(winning_trades)
            max_win = max(t['profit'] for t in winning_trades)
            avg_win_pct = sum(t['profit_pct'] for t in winning_trades) / len(winning_trades)
            
            print(f"\n✅ WINNING TRADES")
            print(f"-" * 60)
            print(f"Total Profit: ${total_wins:.2f}")
            print(f"Average Win: ${avg_win:.2f} ({avg_win_pct:.2f}%)")
            print(f"Largest Win: ${max_win:.2f}")
        
        if losing_trades:
            total_losses = sum(t['profit'] for t in losing_trades)
            avg_loss = total_losses / len(losing_trades)
            max_loss = min(t['profit'] for t in losing_trades)
            avg_loss_pct = sum(t['profit_pct'] for t in losing_trades) / len(losing_trades)
            
            print(f"\n❌ LOSING TRADES")
            print(f"-" * 60)
            print(f"Total Loss: ${total_losses:.2f}")
            print(f"Average Loss: ${avg_loss:.2f} ({avg_loss_pct:.2f}%)")
            print(f"Largest Loss: ${max_loss:.2f}")
        
        # Risk metrics
        if winning_trades and losing_trades:
            avg_win = sum(t['profit'] for t in winning_trades) / len(winning_trades)
            avg_loss = abs(sum(t['profit'] for t in losing_trades) / len(losing_trades))
            profit_factor = abs(sum(t['profit'] for t in winning_trades) / sum(t['profit'] for t in losing_trades))
            
            print(f"\n⚖️  RISK METRICS")
            print(f"-" * 60)
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Risk/Reward Ratio: {avg_win/avg_loss:.2f}")
        
        # Trade list
        print(f"\n📋 TRADE HISTORY")
        print(f"-" * 60)
        for i, trade in enumerate(self.trades, 1):
            status = "✅" if trade['profit'] > 0 else "❌"
            print(f"{status} Trade {i}: Entry ${trade['entry_price']:.2f} → "
                  f"Exit ${trade['exit_price']:.2f} | "
                  f"P/L: ${trade['profit']:+.2f} ({trade['profit_pct']:+.2f}%)")
        
        print("="*60)
    
    def plot_results(self, df):
        """Plot backtest results"""
        if not self.equity_curve:
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            equity_df = pd.DataFrame(self.equity_curve)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # Plot price and trades
            ax1.plot(df['timestamp'], df['close'], label='Price', linewidth=1.5, color='blue')
            
            for trade in self.trades:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                
                # Find closest timestamps in dataframe
                entry_idx = df.iloc[(df['timestamp'] - entry_time).abs().argsort()[:1]].index[0]
                exit_idx = df.iloc[(df['timestamp'] - exit_time).abs().argsort()[:1]].index[0]
                
                color = 'green' if trade['profit'] > 0 else 'red'
                ax1.scatter(df.loc[entry_idx, 'timestamp'], trade['entry_price'], 
                           marker='^', color='green', s=100, zorder=5)
                ax1.scatter(df.loc[exit_idx, 'timestamp'], trade['exit_price'], 
                           marker='v', color=color, s=100, zorder=5)
            
            ax1.set_ylabel('Price (USD)', fontsize=12)
            ax1.set_title(f'{self.symbol} Backtest Results', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot equity curve
            ax2.plot(equity_df['timestamp'], equity_df['equity'], 
                    label='Portfolio Value', linewidth=2, color='purple')
            ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                       label='Initial Capital', alpha=0.7)
            ax2.fill_between(equity_df['timestamp'], self.initial_capital, equity_df['equity'],
                            where=(equity_df['equity'] >= self.initial_capital), 
                            color='green', alpha=0.3)
            ax2.fill_between(equity_df['timestamp'], self.initial_capital, equity_df['equity'],
                            where=(equity_df['equity'] < self.initial_capital), 
                            color='red', alpha=0.3)
            
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Portfolio Value (USD)', fontsize=12)
            ax2.set_title('Equity Curve', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"\n📊 Chart saved as: {filename}")
            
        except ImportError:
            print("\n⚠️  matplotlib not available, skipping chart generation")
        except Exception as e:
            print(f"\n⚠️  Could not generate chart: {e}")


if __name__ == "__main__":
    backtester = Backtester()
    backtester.backtest(days=30)
