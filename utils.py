import yfinance as yf
from typing import Dict, Any, List
from datetime import datetime
from config import TRADING_SETTINGS
import logging
import os
import pandas as pd

def get_market_data(symbols: List[str]) -> Dict[str, Any]:
    """Fetch market data for given symbols"""
    market_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=TRADING_SETTINGS["analysis_timeframe"])
            
            # Add required fields for reflection agent
            market_data[symbol] = {
                # Historical data series
                "prices": hist["Close"].tolist(),
                "volumes": hist["Volume"].tolist(),
                "dates": [d.strftime('%Y-%m-%d') for d in hist.index],
                
                # Current snapshot
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
    """
    Load historical trading decisions and results from all CSV files in evaluation_results directory.
    
    Returns:
        List of dictionaries containing historical decisions with their results
    """
    historical_decisions = []
    eval_dir = "evaluation_results"
    
    # Create directory if it doesn't exist
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
        return historical_decisions
    
    # Find all CSV files in the directory
    csv_files = [f for f in os.listdir(eval_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(os.path.join(eval_dir, csv_file))
            
            # Convert each row to a dictionary with relevant information
            for _, row in df.iterrows():
                decision = {
                    'timestamp': row['Timestamp'],
                    'symbol': row['Symbol'],
                    'predicted_signal': row['Predicted_Signal'],  # True for bullish, False for bearish
                    'actual_return': row['Actual_Return'],
                    'confidence_score': row['Confidence_Score'],
                    'success': row['Binary_Accuracy'] == 1.0,  # Whether prediction was correct
                    'weighted_accuracy': row['Weighted_Accuracy'],  # Accuracy weighted by return magnitude
                    'confidence_weighted_accuracy': row['Confidence_Weighted_Accuracy'],  # Accuracy weighted by confidence
                    'signal_strength': row['Signal_Strength'],
                    'return_direction': row['Return_Direction'],
                    'return_magnitude': row['Return_Magnitude'],
                    'consensus_score': row['Consensus_Score'],
                    'argument_quality': row['Argument_Quality']
                }
                historical_decisions.append(decision)
                
        except Exception as e:
            logging.error(f"Error loading historical decisions from {csv_file}: {str(e)}")
            continue
    
    # Sort by timestamp
    historical_decisions.sort(key=lambda x: datetime.fromisoformat(x['timestamp']))
    
    return historical_decisions