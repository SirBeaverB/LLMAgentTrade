import os
from datetime import datetime
from agents.coordinator_agent import CoordinatorAgent
from config import AGENT_SETTINGS, TRADING_SETTINGS
from utils import get_market_data, load_historical_decisions

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

        print("\nSymbol Signals:")
        print(result['final_decision']['symbol_signals'])
        
        # Print the market context
        print("\nMarket Context:")
        print(result['final_decision']['market_context'])
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 