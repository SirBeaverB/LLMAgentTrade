import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from agents.coordinator_agent import CoordinatorAgent
from config import AGENT_SETTINGS, TRADING_SETTINGS
from utils import get_market_data, load_historical_decisions
import yfinance as yf

class SystemEvaluator:
    def __init__(self):
        self.coordinator = CoordinatorAgent(AGENT_SETTINGS["coordinator_agent"])
        
    def evaluate_signal_accuracy(self, symbols: List[str], lookback_days: int = 30) -> Dict[str, float]:
        """Evaluate the accuracy of trading signals against actual price movements"""
        results = {}
        
        for symbol in symbols:
            # Get historical data
            market_data = {symbol: get_market_data([symbol])[symbol]}
            
            # Generate signals
            analysis = self.coordinator.analyze({
                "symbols": [symbol],
                "market_data": market_data,
                "timestamp": datetime.now().isoformat(),
                "proposed_action": {"type": "ANALYSIS", "symbols": [symbol]}
            })
            
            signal = analysis['final_decision']['symbol_signals'].get(symbol, False)
            
            # Compare with actual price movement
            ticker_data = yf.Ticker(symbol).history(period=f"{lookback_days}d")
            actual_return = (ticker_data['Close'][-1] - ticker_data['Close'][0]) / ticker_data['Close'][0]
            
            # Calculate accuracy (signal matches direction)
            signal_correct = (signal and actual_return > 0) or (not signal and actual_return < 0)
            results[symbol] = 1.0 if signal_correct else 0.0
            
        return results

    def evaluate_debate_quality(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Evaluate the quality of the debate process"""
        results = {}
        
        for symbol in symbols:
            market_data = {symbol: get_market_data([symbol])[symbol]}
            
            # Run analysis
            analysis = self.coordinator.analyze({
                "symbols": [symbol],
                "market_data": market_data,
                "timestamp": datetime.now().isoformat(),
                "proposed_action": {"type": "ANALYSIS", "symbols": [symbol]}
            })
            
            # Extract debate metrics
            debate_rounds = analysis.get('debate_rounds', [])
            
            # Calculate metrics
            metrics = {
                'consensus_score': self._calculate_consensus(debate_rounds),
                'argument_quality': self._evaluate_argument_quality(debate_rounds),
                'perspective_diversity': self._calculate_perspective_diversity(debate_rounds)
            }
            
            results[symbol] = metrics
            
        return results
    
    def evaluate_system_performance(self, symbols: List[str]) -> Dict[str, Any]:
        """Evaluate overall system performance metrics"""
        start_time = datetime.now()
        
        # Run batch analysis
        market_data = get_market_data(symbols)
        analysis = self.coordinator.analyze({
            "symbols": symbols,
            "market_data": market_data,
            "timestamp": start_time.isoformat(),
            "proposed_action": {"type": "ANALYSIS", "symbols": symbols}
        })
        
        end_time = datetime.now()
        
        return {
            'response_time': (end_time - start_time).total_seconds(),
            'confidence_score': analysis['confidence_score'],
            'memory_usage': self._get_memory_usage(),
            'api_calls': self._count_api_calls(analysis)
        }
    
    def evaluate_technical_reliability(self, symbols: List[str], num_iterations: int = 10) -> Dict[str, Any]:
        """Evaluate technical reliability of the system"""
        errors = []
        response_times = []
        success_rate = 0
        
        for _ in range(num_iterations):
            try:
                start_time = datetime.now()
                market_data = get_market_data(symbols)
                analysis = self.coordinator.analyze({
                    "symbols": symbols,
                    "market_data": market_data,
                    "timestamp": datetime.now().isoformat(),
                    "proposed_action": {"type": "ANALYSIS", "symbols": symbols}
                })
                response_times.append((datetime.now() - start_time).total_seconds())
                success_rate += 1
            except Exception as e:
                errors.append(str(e))
        
        return {
            'success_rate': success_rate / num_iterations,
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'error_types': list(set(errors)),
            'stability_score': self._calculate_stability_score(response_times, errors)
        }
    
    def _calculate_consensus(self, debate_rounds: List[Dict[str, Any]]) -> float:
        """Calculate consensus score from debate rounds"""
        if not debate_rounds:
            return 0.0
            
        # Extract all stances
        stances = []
        for round_data in debate_rounds:
            arguments = round_data['arguments'].lower()
            if 'bullish' in arguments:
                stances.append(1)
            elif 'bearish' in arguments:
                stances.append(-1)
                
        if not stances:
            return 0.0
            
        # Calculate consensus as the absolute mean of stances
        return abs(np.mean(stances))
    
    def _evaluate_argument_quality(self, debate_rounds: List[Dict[str, Any]]) -> float:
        """Evaluate the quality of arguments in debate rounds"""
        if not debate_rounds:
            return 0.0
            
        quality_scores = []
        for round_data in debate_rounds:
            arguments = round_data['arguments']
            # Score based on presence of key elements
            score = 0
            if any(term in arguments.lower() for term in ['because', 'due to', 'as a result']):
                score += 0.3  # Reasoning
            if any(term in arguments.lower() for term in ['data', 'numbers', 'statistics', '%']):
                score += 0.3  # Data support
            if any(term in arguments.lower() for term in ['however', 'although', 'while']):
                score += 0.2  # Balanced view
            if len(arguments.split()) >= 20:
                score += 0.2  # Sufficient detail
                
            quality_scores.append(score)
            
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_perspective_diversity(self, debate_rounds: List[Dict[str, Any]]) -> float:
        """Calculate the diversity of perspectives in the debate"""
        if not debate_rounds:
            return 0.0
            
        perspectives = set()
        for round_data in debate_rounds:
            perspectives.add(round_data['perspective'])
            
        return len(perspectives) / 3.0  # Normalized by expected number of perspectives
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage of the system"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def _count_api_calls(self, analysis: Dict[str, Any]) -> int:
        """Count the number of API calls made during analysis"""
        # This would need to be implemented based on your API tracking mechanism
        return len(analysis.get('debate_rounds', [])) + 1  # Debates + final synthesis
    
    def _calculate_stability_score(self, response_times: List[float], errors: List[str]) -> float:
        """Calculate system stability score"""
        if not response_times:
            return 0.0
            
        # Normalize response times
        mean_time = np.mean(response_times)
        std_time = np.std(response_times) if len(response_times) > 1 else 0
        time_stability = 1.0 / (1.0 + std_time/mean_time) if mean_time > 0 else 0
        
        # Error penalty
        error_penalty = len(errors) / (len(response_times) + len(errors))
        
        return (time_stability * (1 - error_penalty))

def main():
    # Initialize evaluator
    evaluator = SystemEvaluator()
    
    # Define test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    print("=== Multi-Agent Trading System Evaluation ===\n")
    
    # 1. Signal Accuracy
    print("1. Signal Accuracy Evaluation")
    signal_accuracy = evaluator.evaluate_signal_accuracy(test_symbols)
    print(f"Average Signal Accuracy: {np.mean(list(signal_accuracy.values())):.2%}")
    for symbol, accuracy in signal_accuracy.items():
        print(f"{symbol}: {accuracy:.2%}")
    print()
    
    # 2. Debate Quality
    print("2. Debate Quality Evaluation")
    debate_quality = evaluator.evaluate_debate_quality(test_symbols)
    for symbol, metrics in debate_quality.items():
        print(f"\n{symbol}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
    print()
    
    # 3. System Performance
    print("3. System Performance Evaluation")
    performance = evaluator.evaluate_system_performance(test_symbols)
    for metric, value in performance.items():
        print(f"{metric}: {value}")
    print()
    
    # 4. Technical Reliability
    print("4. Technical Reliability Evaluation")
    reliability = evaluator.evaluate_technical_reliability(test_symbols)
    for metric, value in reliability.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main() 