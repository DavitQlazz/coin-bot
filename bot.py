import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import os
import ta

class CryptoTradingBot:
    def __init__(self):
        load_dotenv()
        self.exchange = self._initialize_exchange()
        self.symbol = os.getenv('TRADING_PAIR', 'BTC/USDT')
        self.timeframe = os.getenv('TIMEFRAME', '1h')
        self.capital = float(os.getenv('INITIAL_CAPITAL', 1000))
        self.initial_capital = self.capital
        self.position = None
        self.trades = []
        self.paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PERCENT', 2.0))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PERCENT', 5.0))
        
    def _initialize_exchange(self):
        """Initialize exchange connection"""
        exchange_id = os.getenv('EXCHANGE', 'binance')
        exchange_class = getattr(ccxt, exchange_id)
        
        exchange = exchange_class({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('API_SECRET'),
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}  # Use 'future' for futures trading
        })
        
        # Test connection
        try:
            exchange.load_markets()
            print(f"✅ Connected to {exchange_id.upper()}")
        except Exception as e:
            print(f"⚠️  Connection error: {e}")
            print("Running in demo mode with limited functionality")
        
        return exchange
    
    def fetch_ohlcv(self, limit=100):
        """Fetch historical price data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        
        # EMA (Exponential Moving Average)
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        # ATR (Average True Range) for volatility
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        return df
    
    def generate_signal(self, df):
        """Generate trading signals based on multiple indicators"""
        if len(df) < 50:
            return 'HOLD'
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Multi-indicator strategy
        # BUY conditions:
        # 1. RSI oversold but recovering
        # 2. MACD bullish crossover
        # 3. EMA golden cross (short-term above long-term)
        # 4. Price near lower Bollinger Band
        
        buy_conditions = [
            latest['rsi'] < 35 and latest['rsi'] > previous['rsi'],  # RSI recovering from oversold
            latest['macd'] > latest['macd_signal'] and previous['macd'] <= previous['macd_signal'],  # MACD crossover
            latest['ema_9'] > latest['ema_21'],  # Short EMA above long EMA
            latest['close'] < latest['bb_mid']  # Price below middle band
        ]
        
        # SELL conditions:
        # 1. RSI overbought
        # 2. MACD bearish crossover
        # 3. EMA death cross
        # 4. Price near upper Bollinger Band
        
        sell_conditions = [
            latest['rsi'] > 70,  # Overbought
            latest['macd'] < latest['macd_signal'] and previous['macd'] >= previous['macd_signal'],  # MACD bearish
            latest['ema_9'] < latest['ema_21'],  # Death cross
            latest['close'] > latest['bb_high']  # Price above upper band
        ]
        
        # Check stop loss and take profit if in position
        if self.position:
            profit_pct = ((latest['close'] / self.position['entry_price']) - 1) * 100
            
            if profit_pct <= -self.stop_loss_pct:
                print(f"🛑 Stop Loss triggered: {profit_pct:.2f}%")
                return 'SELL'
            
            if profit_pct >= self.take_profit_pct:
                print(f"🎯 Take Profit triggered: {profit_pct:.2f}%")
                return 'SELL'
        
        # Generate signal based on conditions
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        if buy_score >= 3 and not self.position:
            return 'BUY'
        elif (sell_score >= 2 or sell_conditions[0]) and self.position:  # Sell if overbought or 2+ conditions
            return 'SELL'
        
        return 'HOLD'
    
    def execute_trade(self, signal, current_price):
        """Execute trades based on signals"""
        timestamp = datetime.now()
        
        if signal == 'BUY' and not self.position:
            # Calculate position size
            position_size_pct = float(os.getenv('POSITION_SIZE_PERCENT', 95)) / 100
            amount = (self.capital * position_size_pct) / current_price
            
            print(f"\n{'='*60}")
            print(f"🟢 BUY SIGNAL TRIGGERED")
            print(f"{'='*60}")
            print(f"Symbol: {self.symbol}")
            print(f"Price: ${current_price:.2f}")
            print(f"Amount: {amount:.6f}")
            print(f"Position Size: ${self.capital * position_size_pct:.2f}")
            print(f"Mode: {'PAPER TRADING' if self.paper_trading else 'LIVE TRADING'}")
            
            if not self.paper_trading:
                try:
                    order = self.exchange.create_market_buy_order(self.symbol, amount)
                    print(f"✅ Order executed: {order['id']}")
                except Exception as e:
                    print(f"❌ Order failed: {e}")
                    return
            
            self.position = {
                'entry_price': current_price,
                'amount': amount,
                'timestamp': timestamp,
                'type': 'LONG'
            }
            
            print(f"{'='*60}\n")
            
        elif signal == 'SELL' and self.position:
            profit = (current_price - self.position['entry_price']) * self.position['amount']
            profit_pct = ((current_price / self.position['entry_price']) - 1) * 100
            
            print(f"\n{'='*60}")
            print(f"🔴 SELL SIGNAL TRIGGERED")
            print(f"{'='*60}")
            print(f"Symbol: {self.symbol}")
            print(f"Entry Price: ${self.position['entry_price']:.2f}")
            print(f"Exit Price: ${current_price:.2f}")
            print(f"Profit/Loss: ${profit:.2f} ({profit_pct:+.2f}%)")
            print(f"Mode: {'PAPER TRADING' if self.paper_trading else 'LIVE TRADING'}")
            
            if not self.paper_trading:
                try:
                    order = self.exchange.create_market_sell_order(self.symbol, self.position['amount'])
                    print(f"✅ Order executed: {order['id']}")
                except Exception as e:
                    print(f"❌ Order failed: {e}")
                    return
            
            self.capital += profit
            
            trade_record = {
                'entry_price': self.position['entry_price'],
                'exit_price': current_price,
                'amount': self.position['amount'],
                'profit': profit,
                'profit_pct': profit_pct,
                'entry_time': self.position['timestamp'].isoformat(),
                'exit_time': timestamp.isoformat(),
                'duration': str(timestamp - self.position['timestamp'])
            }
            
            self.trades.append(trade_record)
            self.position = None
            
            # Save trades to file
            self.save_trades()
            
            print(f"{'='*60}\n")
    
    def save_trades(self):
        """Save trade history to JSON file"""
        try:
            with open('trades.json', 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save trades: {e}")
    
    def run(self):
        """Main bot loop"""
        print(f"\n{'='*60}")
        print(f"🤖 CRYPTO TRADING BOT STARTED")
        print(f"{'='*60}")
        print(f"📊 Symbol: {self.symbol}")
        print(f"⏰ Timeframe: {self.timeframe}")
        print(f"💰 Initial Capital: ${self.initial_capital:.2f}")
        print(f"🎯 Stop Loss: {self.stop_loss_pct}%")
        print(f"🎯 Take Profit: {self.take_profit_pct}%")
        print(f"📝 Mode: {'PAPER TRADING' if self.paper_trading else 'LIVE TRADING'}")
        print(f"{'='*60}\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                
                # Fetch and analyze data
                df = self.fetch_ohlcv()
                
                if df is None or len(df) == 0:
                    print("⚠️  No data received, retrying in 60 seconds...")
                    time.sleep(60)
                    continue
                
                df = self.calculate_indicators(df)
                signal = self.generate_signal(df)
                current_price = df.iloc[-1]['close']
                
                # Display status
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                status = f"[{timestamp}] Price: ${current_price:,.2f} | Signal: {signal:4s} | RSI: {df.iloc[-1]['rsi']:.1f}"
                
                if self.position:
                    unrealized_pnl = ((current_price / self.position['entry_price']) - 1) * 100
                    status += f" | Position: LONG | PnL: {unrealized_pnl:+.2f}%"
                
                print(status)
                
                # Execute trade if signal generated
                self.execute_trade(signal, current_price)
                
                # Print summary every 10 iterations
                if iteration % 10 == 0 and self.trades:
                    self.print_quick_stats()
                
                # Wait before next iteration
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\n⚠️  Bot stopped by user")
            self.print_summary()
    
    def print_quick_stats(self):
        """Print quick statistics"""
        if not self.trades:
            return
        
        total_profit = sum(t['profit'] for t in self.trades)
        win_rate = len([t for t in self.trades if t['profit'] > 0]) / len(self.trades) * 100
        roi = ((self.capital / self.initial_capital) - 1) * 100
        
        print(f"\n📊 Stats: Trades: {len(self.trades)} | Win Rate: {win_rate:.1f}% | "
              f"Total P/L: ${total_profit:+.2f} | ROI: {roi:+.2f}%\n")
    
    def print_summary(self):
        """Print comprehensive trading summary"""
        print("\n" + "="*60)
        print("📊 TRADING SESSION SUMMARY")
        print("="*60)
        
        if not self.trades:
            print("No trades executed during this session.")
            print("="*60)
            return
        
        total_profit = sum(t['profit'] for t in self.trades)
        winning_trades = [t for t in self.trades if t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['profit'] < 0]
        
        avg_profit = total_profit / len(self.trades)
        win_rate = len(winning_trades) / len(self.trades) * 100
        roi = ((self.capital / self.initial_capital) - 1) * 100
        
        print(f"Total Trades: {len(self.trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"\nInitial Capital: ${self.initial_capital:.2f}")
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total Profit/Loss: ${total_profit:+.2f}")
        print(f"Average P/L per Trade: ${avg_profit:+.2f}")
        print(f"ROI: {roi:+.2f}%")
        
        if winning_trades:
            avg_win = sum(t['profit'] for t in winning_trades) / len(winning_trades)
            max_win = max(t['profit'] for t in winning_trades)
            print(f"\nAverage Win: ${avg_win:.2f}")
            print(f"Largest Win: ${max_win:.2f}")
        
        if losing_trades:
            avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades)
            max_loss = min(t['profit'] for t in losing_trades)
            print(f"Average Loss: ${avg_loss:.2f}")
            print(f"Largest Loss: ${max_loss:.2f}")
        
        print("="*60)


if __name__ == "__main__":
    bot = CryptoTradingBot()
    bot.run()
