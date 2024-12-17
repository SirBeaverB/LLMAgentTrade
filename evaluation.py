import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from agents.coordinator_agent import CoordinatorAgent
from config import AGENT_SETTINGS, TRADING_SETTINGS
from utils import get_market_data, load_historical_decisions
import yfinance as yf
import random
import logging
import argparse

# Set up logging
def setup_logging(log_dir: str = "logs") -> str:
    """Set up logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return log_file

class SystemEvaluator:
    def __init__(self, random_seed: int = 42):
        self.coordinator = CoordinatorAgent(AGENT_SETTINGS["coordinator_agent"])
        random.seed(random_seed)
        np.random.seed(random_seed)
        logging.info(f"Initialized SystemEvaluator with random seed: {random_seed}")
    
    def run_comprehensive_evaluation(self, num_symbols: int = 100, lookback_days: int = 30) -> Dict[str, Any]:
        """Run a comprehensive evaluation across all dimensions for the specified number of symbols"""
        logging.info(f"Starting comprehensive evaluation with {num_symbols} symbols, lookback days: {lookback_days}")
        
        # Get list of valid symbols
        with open('utils/symbols.txt', 'r') as f:
            all_symbols = [line.strip() for line in f if line.strip()]
        
        # Select random symbols
        selected_symbols = random.sample(all_symbols, min(num_symbols, len(all_symbols)))
        logging.info(f"Selected symbols: {', '.join(selected_symbols)}")
        
        results = {
            'signal_accuracy': {},
            'debate_quality': {},
            'system_performance': {},
            'technical_reliability': {},
            'overall_metrics': {},
            'evaluation_params': {
                'num_symbols': num_symbols,
                'lookback_days': lookback_days,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Individual symbol evaluation
        for i, symbol in enumerate(selected_symbols, 1):
            logging.info(f"\nEvaluating symbol {i}/{num_symbols}: {symbol}")
            try:
                # Get historical data and generate analysis once
                ticker = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days + 1)  # +1 for validation day
                
                # Get historical data
                hist_data = ticker.history(start=start_date, end=end_date)
                if len(hist_data) < 2:  # Need at least 2 days of data
                    logging.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Prepare training data (everything except last day)
                training_data = hist_data[:-1]
                
                # Prepare market data format
                market_data = {
                    symbol: {
                        'prices': training_data['Close'].tolist(),
                        'volumes': training_data['Volume'].tolist(),
                        'high': training_data['High'].tolist(),
                        'low': training_data['Low'].tolist(),
                        'dates': [d.strftime('%Y-%m-%d') for d in training_data.index]
                    }
                }
                
                # Generate analysis using historical data
                start_time = datetime.now()
                analysis = self.coordinator.analyze({
                    "symbols": [symbol],
                    "market_data": market_data,
                    "timestamp": training_data.index[-1].isoformat(),
                    "proposed_action": {"type": "ANALYSIS", "symbols": [symbol]}
                })
                end_time = datetime.now()
                
                # 1. Signal Accuracy
                validation_data = hist_data[-1:]
                if len(validation_data) > 0:
                    actual_return = (validation_data['Close'].iloc[-1] - training_data['Close'].iloc[-1]) / training_data['Close'].iloc[-1]
                    signal = analysis['final_decision']['symbol_signals'].get(symbol, False)
                    confidence = analysis.get('confidence_score', 0.5)  # Default to 0.5 if not provided
                    
                    # Binary accuracy (1.0 for correct, 0.0 for wrong)
                    signal_correct = (signal and actual_return > 0) or (not signal and actual_return < 0)
                    binary_accuracy = 1.0 if signal_correct else 0.0
                    
                    # Weighted accuracy based on return magnitude
                    abs_return = abs(actual_return)
                    weighted_accuracy = abs_return if signal_correct else -abs_return
                    
                    # Confidence-weighted accuracy
                    # Scale from [-confidence, +confidence] based on correctness
                    confidence_weighted = confidence if signal_correct else -confidence
                    
                    # Combined accuracy (confidence * return magnitude)
                    # This will give higher scores for being confident AND right about big moves
                    # And bigger penalties for being confident AND wrong about big moves
                    combined_weighted = confidence * weighted_accuracy
                    
                    # Store all metrics
                    results['signal_accuracy'][symbol] = {
                        'binary': binary_accuracy,
                        'weighted': weighted_accuracy,
                        'confidence': confidence,
                        'confidence_weighted': confidence_weighted,
                        'combined_weighted': combined_weighted,
                        'actual_return': actual_return,
                        'predicted_signal': signal
                    }
                    
                    logging.info(f"Signal: {'BULLISH' if signal else 'BEARISH'}, "
                          f"Confidence: {confidence:.2%}, "
                          f"Actual Return: {actual_return:.2%}, "
                          f"Binary Accuracy: {binary_accuracy:.2f}, "
                          f"Weighted Accuracy: {weighted_accuracy:+.2%}, "
                          f"Confidence-Weighted: {confidence_weighted:+.2%}, "
                          f"Combined-Weighted: {combined_weighted:+.2%}")
                
                # 2. Debate Quality
                debate_analysis = analysis.get('component_analyses', {}).get('debate', {})
                debate_rounds = debate_analysis.get('debate_rounds', [])
                results['debate_quality'][symbol] = {
                    'consensus_score': self._calculate_consensus(debate_rounds),
                    'argument_quality': self._evaluate_argument_quality(debate_rounds),
                    'perspective_diversity': self._calculate_perspective_diversity(debate_rounds)
                }
                logging.info(f"Debate rounds analyzed: {len(debate_rounds)}")
                
                # 3. System Performance
                results['system_performance'][symbol] = {
                    'response_time': (end_time - start_time).total_seconds(),
                    'confidence_score': analysis.get('confidence_score', 0),
                    'memory_usage': self._get_memory_usage(),
                    'api_calls': self._count_api_calls(analysis)
                }
                logging.info(f"Response time: {(end_time - start_time).total_seconds():.2f}s")
                
                # 4. Technical Reliability
                error_count = 0
                response_times = []
                for _ in range(3):  # Run 3 iterations for stability check
                    try:
                        iter_start = datetime.now()
                        # Just check if we can parse the existing analysis
                        _ = analysis['final_decision']['symbol_signals'].get(symbol, False)
                        response_times.append((datetime.now() - iter_start).total_seconds())
                    except Exception as e:
                        error_count += 1
                        logging.error(f"Error in reliability check: {str(e)}")
                
                results['technical_reliability'][symbol] = {
                    'success_rate': (3 - error_count) / 3,
                    'stability_score': self._calculate_stability_score(response_times, []),
                    'error_types': []
                }
                
                logging.info(f"✓ {symbol} evaluation complete")
                
            except Exception as e:
                logging.error(f"✗ Error evaluating {symbol}: {str(e)}", exc_info=True)
                results['signal_accuracy'][symbol] = 0.0
                results['debate_quality'][symbol] = {'consensus_score': 0.0, 'argument_quality': 0.0, 'perspective_diversity': 0.0}
                results['system_performance'][symbol] = {'response_time': 0.0, 'confidence_score': 0.0}
                results['technical_reliability'][symbol] = {'success_rate': 0.0, 'stability_score': 0.0}
        
        # Calculate overall metrics
        self._calculate_overall_metrics(results, num_symbols)
        
        # Print and store summary
        print_summary(results)
        
        # Save results (moved after summary is added)
        self._save_results(results)
        
        return results
    
    def _calculate_overall_metrics(self, results: Dict[str, Any], num_symbols: int):
        """Calculate overall metrics from individual results"""
        # Get valid results (exclude failed evaluations)
        valid_signals = [v for v in results['signal_accuracy'].values() if isinstance(v, dict) and v.get('binary') is not None]
        valid_debates = [m for m in results['debate_quality'].values() if all(v is not None for v in m.values())]
        valid_performance = [m for m in results['system_performance'].values() if all(v is not None for v in m.values())]
        valid_reliability = [m for m in results['technical_reliability'].values() if all(v is not None for v in m.values())]
        
        num_valid = len(valid_signals)
        logging.info(f"Valid evaluations: {num_valid}/{num_symbols}")
        
        # Calculate signal accuracy metrics
        binary_accuracies = [v['binary'] for v in valid_signals]
        weighted_accuracies = [v['weighted'] for v in valid_signals]
        confidence_weighted = [v['confidence_weighted'] for v in valid_signals]
        combined_weighted = [v['combined_weighted'] for v in valid_signals]
        actual_returns = [v['actual_return'] for v in valid_signals]
        confidences = [v['confidence'] for v in valid_signals]
        
        # Calculate additional metrics
        correct_predictions = sum(1 for v in valid_signals if v['binary'] > 0.5)
        high_confidence_correct = sum(1 for v in valid_signals if v['binary'] > 0.5 and v['confidence'] > 0.7)
        total_return = sum(weighted_accuracies)
        avg_magnitude = np.mean([abs(r) for r in actual_returns]) if actual_returns else 0.0
        
        results['overall_metrics'] = {
            'signal_accuracy': {
                'binary_mean': np.mean(binary_accuracies) if binary_accuracies else 0.0,
                'binary_std': np.std(binary_accuracies) if binary_accuracies else 0.0,
                'weighted_mean': np.mean(weighted_accuracies) if weighted_accuracies else 0.0,
                'weighted_std': np.std(weighted_accuracies) if weighted_accuracies else 0.0,
                'confidence_weighted_mean': np.mean(confidence_weighted) if confidence_weighted else 0.0,
                'confidence_weighted_std': np.std(confidence_weighted) if confidence_weighted else 0.0,
                'combined_weighted_mean': np.mean(combined_weighted) if combined_weighted else 0.0,
                'combined_weighted_std': np.std(combined_weighted) if combined_weighted else 0.0,
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'high_confidence_correct': high_confidence_correct,
                'total_return': total_return if weighted_accuracies else 0.0,
                'avg_return_magnitude': avg_magnitude,
                'num_correct': correct_predictions,
                'num_valid': num_valid,
                'success_rate': correct_predictions / num_valid if num_valid > 0 else 0.0
            },
            'debate_quality': {
                'mean_consensus': np.mean([m['consensus_score'] for m in valid_debates]) if valid_debates else 0.0,
                'mean_argument_quality': np.mean([m['argument_quality'] for m in valid_debates]) if valid_debates else 0.0,
                'mean_diversity': np.mean([m['perspective_diversity'] for m in valid_debates]) if valid_debates else 0.0
            },
            'system_performance': {
                'mean_response_time': np.mean([m['response_time'] for m in valid_performance]) if valid_performance else 0.0,
                'mean_confidence': np.mean([m.get('confidence_score', 0) for m in valid_performance]) if valid_performance else 0.0,
                'total_api_calls': sum(m.get('api_calls', 0) for m in valid_performance)
            },
            'technical_reliability': {
                'overall_success_rate': np.mean([m['success_rate'] for m in valid_reliability]) if valid_reliability else 0.0,
                'mean_stability': np.mean([m['stability_score'] for m in valid_reliability]) if valid_reliability else 0.0,
                'error_types': list(set(sum([m.get('error_types', []) for m in results['technical_reliability'].values()], [])))
            }
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as CSV
        results_df = pd.DataFrame({
            'Symbol': list(results['signal_accuracy'].keys()),
            'Signal_Accuracy': list(results['signal_accuracy'].values()),
            'Consensus_Score': [m['consensus_score'] for m in results['debate_quality'].values()],
            'Argument_Quality': [m['argument_quality'] for m in results['debate_quality'].values()],
            'Perspective_Diversity': [m['perspective_diversity'] for m in results['debate_quality'].values()],
            'Response_Time': [m['response_time'] for m in results['system_performance'].values()],
            'Success_Rate': [m['success_rate'] for m in results['technical_reliability'].values()]
        })
        
        # Create results directory if it doesn't exist
        results_dir = "evaluation_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save CSV and full results
        csv_path = os.path.join(results_dir, f'evaluation_results_{timestamp}.csv')
        json_path = os.path.join(results_dir, f'evaluation_results_{timestamp}.json')
        
        results_df.to_csv(csv_path, index=False)
        pd.DataFrame([results]).to_json(json_path, orient='records')
        
        logging.info(f"Results saved to:\n- {csv_path}\n- {json_path}")
    
    def evaluate_signal_accuracy(self, symbols: List[str], lookback_days: int = 30) -> Dict[str, float]:
        """
        Evaluate the accuracy of trading signals against actual price movements.
        Uses historical data up to today to generate signal, then validates against next day's movement.
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Get historical data up to yesterday
                ticker = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days + 1)  # +1 for validation day
                
                # Get historical data
                hist_data = ticker.history(start=start_date, end=end_date)
                if len(hist_data) < 2:  # Need at least 2 days of data
                    print(f"Insufficient data for {symbol}")
                    results[symbol] = 0.0
                    continue
                
                # Prepare training data (everything except last day)
                training_data = hist_data[:-1]
                
                # Prepare market data format
                market_data = {
                    symbol: {
                        'prices': training_data['Close'].tolist(),
                        'volumes': training_data['Volume'].tolist(),
                        'high': training_data['High'].tolist(),
                        'low': training_data['Low'].tolist(),
                        'dates': [d.strftime('%Y-%m-%d') for d in training_data.index]
                    }
                }
                
                # Generate signal using historical data
                analysis = self.coordinator.analyze({
                    "symbols": [symbol],
                    "market_data": market_data,
                    "timestamp": training_data.index[-1].isoformat(),
                    "proposed_action": {"type": "ANALYSIS", "symbols": [symbol]}
                })
                
                signal = analysis['final_decision']['symbol_signals'].get(symbol, False)
                
                # Get actual price movement (last day)
                validation_data = hist_data[-1:]
                if len(validation_data) > 0:
                    actual_return = (validation_data['Close'].iloc[-1] - training_data['Close'].iloc[-1]) / training_data['Close'].iloc[-1]
                    
                    # Calculate accuracy (signal matches direction)
                    signal_correct = (signal and actual_return > 0) or (not signal and actual_return < 0)
                    results[symbol] = 1.0 if signal_correct else 0.0
                    
                    print(f"{symbol} - Signal: {'BULLISH' if signal else 'BEARISH'}, "
                          f"Actual Return: {actual_return:.2%}, Correct: {signal_correct}")
                else:
                    print(f"No validation data available for {symbol}")
                    results[symbol] = 0.0
                    
            except Exception as e:
                print(f"Error evaluating {symbol}: {str(e)}")
                results[symbol] = 0.0
        
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

def print_summary(results: Dict[str, Any]):
    """Print evaluation summary and store it in results dictionary"""
    logging.info("\nEvaluation Summary:")
    logging.info("==================")
    
    num_valid = results['overall_metrics']['signal_accuracy']['num_valid']
    num_correct = results['overall_metrics']['signal_accuracy']['num_correct']
    high_conf_correct = results['overall_metrics']['signal_accuracy']['high_confidence_correct']
    total_symbols = results['evaluation_params']['num_symbols']
    
    # Create summary dictionary
    summary = {
        'evaluation_coverage': {
            'valid_evaluations': num_valid,
            'total_symbols': total_symbols,
            'coverage_rate': num_valid/total_symbols if total_symbols > 0 else 0
        },
        'signal_accuracy': {
            'num_valid_evaluations': num_valid,
            'binary_accuracy': {
                'mean': results['overall_metrics']['signal_accuracy']['binary_mean'],
                'std': results['overall_metrics']['signal_accuracy']['binary_std']
            },
            'weighted_accuracy': {
                'mean': results['overall_metrics']['signal_accuracy']['weighted_mean'],
                'std': results['overall_metrics']['signal_accuracy']['weighted_std']
            },
            'confidence_weighted': {
                'mean': results['overall_metrics']['signal_accuracy']['confidence_weighted_mean'],
                'std': results['overall_metrics']['signal_accuracy']['confidence_weighted_std']
            },
            'combined_weighted': {
                'mean': results['overall_metrics']['signal_accuracy']['combined_weighted_mean'],
                'std': results['overall_metrics']['signal_accuracy']['combined_weighted_std']
            },
            'average_confidence': results['overall_metrics']['signal_accuracy']['avg_confidence'],
            'total_return': results['overall_metrics']['signal_accuracy']['total_return'],
            'average_return_magnitude': results['overall_metrics']['signal_accuracy']['avg_return_magnitude'],
            'correct_predictions': {
                'count': num_correct,
                'total': num_valid,
                'rate': num_correct/num_valid if num_valid > 0 else 0
            },
            'high_confidence_correct': {
                'count': high_conf_correct,
                'total': num_valid,
                'rate': high_conf_correct/num_valid if num_valid > 0 else 0
            }
        },
        'debate_quality': {
            'mean_consensus': results['overall_metrics']['debate_quality']['mean_consensus'],
            'mean_argument_quality': results['overall_metrics']['debate_quality']['mean_argument_quality'],
            'mean_perspective_diversity': results['overall_metrics']['debate_quality']['mean_diversity']
        },
        'system_performance': {
            'mean_response_time': results['overall_metrics']['system_performance']['mean_response_time'],
            'mean_confidence': results['overall_metrics']['system_performance']['mean_confidence'],
            'total_api_calls': results['overall_metrics']['system_performance']['total_api_calls']
        },
        'technical_reliability': {
            'overall_success_rate': results['overall_metrics']['technical_reliability']['overall_success_rate'],
            'mean_stability_score': results['overall_metrics']['technical_reliability']['mean_stability'],
            'error_types': results['overall_metrics']['technical_reliability']['error_types']
        }
    }
    
    # Store summary in results
    results['summary'] = summary
    
    # Print summary
    logging.info(f"\nValid Evaluations: {num_valid}/{total_symbols} ({(num_valid/total_symbols):.1%})")
    
    logging.info(f"\nSignal Accuracy (based on {num_valid} valid evaluations):")
    logging.info(f"Binary Accuracy: {results['overall_metrics']['signal_accuracy']['binary_mean']:.2%} ± {results['overall_metrics']['signal_accuracy']['binary_std']:.2%}")
    logging.info(f"Weighted Accuracy: {results['overall_metrics']['signal_accuracy']['weighted_mean']:+.2%} ± {results['overall_metrics']['signal_accuracy']['weighted_std']:.2%}")
    logging.info(f"Confidence-Weighted: {results['overall_metrics']['signal_accuracy']['confidence_weighted_mean']:+.2%} ± {results['overall_metrics']['signal_accuracy']['confidence_weighted_std']:.2%}")
    logging.info(f"Combined-Weighted: {results['overall_metrics']['signal_accuracy']['combined_weighted_mean']:+.2%} ± {results['overall_metrics']['signal_accuracy']['combined_weighted_std']:.2%}")
    logging.info(f"Average Confidence: {results['overall_metrics']['signal_accuracy']['avg_confidence']:.2%}")
    logging.info(f"Total Return: {results['overall_metrics']['signal_accuracy']['total_return']:+.2%}")
    logging.info(f"Average Return Magnitude: {results['overall_metrics']['signal_accuracy']['avg_return_magnitude']:.2%}")
    logging.info(f"Correct Predictions: {num_correct}/{num_valid} ({(num_correct/num_valid if num_valid > 0 else 0):.1%})")
    logging.info(f"High-Confidence Correct: {high_conf_correct}/{num_valid} ({(high_conf_correct/num_valid if num_valid > 0 else 0):.1%})")
    
    logging.info(f"\nDebate Quality:")
    logging.info(f"Mean consensus: {results['overall_metrics']['debate_quality']['mean_consensus']:.2f}")
    logging.info(f"Mean argument quality: {results['overall_metrics']['debate_quality']['mean_argument_quality']:.2f}")
    logging.info(f"Mean perspective diversity: {results['overall_metrics']['debate_quality']['mean_diversity']:.2f}")
    
    logging.info(f"\nSystem Performance:")
    logging.info(f"Mean response time: {results['overall_metrics']['system_performance']['mean_response_time']:.2f}s")
    logging.info(f"Mean confidence: {results['overall_metrics']['system_performance']['mean_confidence']:.2%}")
    logging.info(f"Total API calls: {results['overall_metrics']['system_performance']['total_api_calls']}")
    
    logging.info(f"\nTechnical Reliability:")
    logging.info(f"Overall success rate: {results['overall_metrics']['technical_reliability']['overall_success_rate']:.2%}")
    logging.info(f"Mean stability score: {results['overall_metrics']['technical_reliability']['mean_stability']:.2f}")
    if results['overall_metrics']['technical_reliability']['error_types']:
        logging.info("Error types encountered:")
        for error in results['overall_metrics']['technical_reliability']['error_types']:
            logging.info(f"- {error}")

def main():
    """Run the evaluation with command line arguments"""
    parser = argparse.ArgumentParser(description='Run comprehensive evaluation of the trading system')
    parser.add_argument('--num_symbols', type=int, default=10,
                      help='Number of symbols to evaluate (default: 10)')
    parser.add_argument('--lookback_days', type=int, default=30,
                      help='Number of days to look back for historical data (default: 30)')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    logging.info(f"Starting evaluation with arguments: {args}")
    
    # Run evaluation
    evaluator = SystemEvaluator(random_seed=args.random_seed)
    results = evaluator.run_comprehensive_evaluation(
        num_symbols=args.num_symbols,
        lookback_days=args.lookback_days
    )
    
    # # Print and log summary
    # print_summary(results)
    logging.info(f"Evaluation complete. Log file: {log_file}")

if __name__ == "__main__":
    main() 