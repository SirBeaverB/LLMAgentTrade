import yfinance as yf
from typing import Dict, Any, List
from datetime import datetime
from config import TRADING_SETTINGS

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