import os
from datetime import datetime
from typing import Dict, Any, List
import yfinance as yf
from agents.coordinator_agent import CoordinatorAgent
from config import AGENT_SETTINGS, TRADING_SETTINGS

def get_market_data(symbols: List[str]) -> Dict[str, Any]:
    """Fetch market data for given symbols"""
    market_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=TRADING_SETTINGS["analysis_timeframe"])
            market_data[symbol] = {
                "current_price": hist["Close"].iloc[-1],
                "open": hist["Open"].iloc[-1],
                "high": hist["High"].iloc[-1],
                "low": hist["Low"].iloc[-1],
                "volume": hist["Volume"].iloc[-1],
                "change_percent": ((hist["Close"].iloc[-1] - hist["Open"].iloc[-1]) / hist["Open"].iloc[-1]) * 100
            }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    return market_data

def load_historical_decisions() -> List[Dict[str, Any]]:
    """Load historical trading decisions"""
    # This would typically load from a database
    # For now, return a sample
    return [
        {
            "timestamp": "2023-12-01T10:00:00",
            "symbol": "AAPL",
            "action": "BUY",
            "price": 190.50,
            "outcome": "success",
            "return": 0.025
        }
    ]

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Initialize the coordinator agent
    coordinator = CoordinatorAgent(AGENT_SETTINGS["coordinator_agent"])
    
    # Define symbols to analyze
    symbols = ["AAPL", "MSFT", "GOOGL"]  # Example symbols
    
    # Prepare data for analysis
    market_data = get_market_data(symbols)
    historical_decisions = load_historical_decisions()
    
    # Create analysis context
    analysis_context = {
        "symbols": symbols,
        "market_data": market_data,
        "historical_decisions": historical_decisions,
        "timestamp": datetime.now().isoformat(),
        "proposed_action": {
            "type": "ANALYSIS",
            "symbols": symbols,
            "risk_level": TRADING_SETTINGS["risk_tolerance"]
        }
    }
    
    # Get trading decision
    try:
        result = coordinator.analyze(analysis_context)
        
        print("\n=== Trading Analysis Results ===")
        print(f"\nTimestamp: {result['timestamp']}")
        print(f"\nConfidence Score: {result['confidence_score']:.2%}")
        print("\nFinal Decision:")
        print(result['final_decision']['decision'])
        
        # You could implement actual trading execution here
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 