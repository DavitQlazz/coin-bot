#!/bin/bash
# Quick Start Script for Crypto Trading Bot

echo "🤖 Crypto Trading Bot - Quick Start"
echo "===================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env configuration file..."
    cp .env.example .env
    echo "✅ .env file created!"
    echo ""
fi

# Show menu
echo "What would you like to do?"
echo ""
echo "1) 🎮 Run Demo (No API keys needed)"
echo "2) 📊 Monitor Market (Read-only, public data)"
echo "3) 🔄 Backtest Strategy (Requires API keys)"
echo "4) 🤖 Run Live Bot (Requires API keys & configuration)"
echo "5) 🔧 Optimize Strategy (Find best parameters)"
echo "6) ⚙️  Install Dependencies"
echo "7) ❌ Exit"
echo ""
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo ""
        echo "🎮 Starting demo with simulated data..."
        python demo.py
        ;;
    2)
        echo ""
        echo "⚠️  Note: Monitoring requires API keys in .env file"
        echo "Press Ctrl+C to stop"
        sleep 2
        python monitor.py
        ;;
    3)
        echo ""
        echo "🔄 Running backtest..."
        python backtest.py
        ;;
    4)
        echo ""
        echo "⚠️  WARNING: This will execute real trades!"
        echo "Make sure PAPER_TRADING=true in .env for practice"
        echo ""
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            python bot.py
        else
            echo "Cancelled."
        fi
        ;;
    5)
        echo ""
        echo "🔧 Starting strategy optimizer..."
        python optimize.py
        ;;
    6)
        echo ""
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
        echo "✅ Installation complete!"
        ;;
    7)
        echo "Goodbye! 👋"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac
