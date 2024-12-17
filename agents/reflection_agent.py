from typing import Dict, Any, List
import pandas as pd
import yfinance as yf
from agents import BaseAgent
from datetime import datetime, timedelta
import pytz
import numpy as np
from collections import defaultdict
import logging


class ReflectionAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.memory = defaultdict(list)  # Initialize memory storage

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and provide reflective insights"""
        # Input validation with detailed logging
        required_fields = {
            "symbols": data.get("symbols", []),
            "timestamp": data.get("timestamp"),
            "market_data": data.get("market_data", {}),
            "news_analysis": data.get("news_analysis", {})
        }
        
        # # Log all input data for debugging
        # logging.info("Reflection Agent received input data:")
        # for field, value in required_fields.items():
        #     logging.info(f"- {field}: {type(value)} - {value if not isinstance(value, dict) else len(value)} items")

        # Validate symbols
        if not required_fields["symbols"]:
            logging.error("No symbols provided for analysis")
            return self._create_error_response(required_fields["timestamp"], "No symbols provided for analysis")

        # Validate current market data
        if not required_fields["market_data"]:
            logging.error("No current market data provided")
            return self._create_error_response(required_fields["timestamp"], "No current market data provided")
            
        # Validate market data structure
        for symbol in required_fields["symbols"]:
            if symbol not in required_fields["market_data"]:
                logging.error(f"Missing market data for symbol: {symbol}")
                return self._create_error_response(required_fields["timestamp"], f"Missing market data for symbol: {symbol}")
            
            market_data = required_fields["market_data"][symbol]
            if not isinstance(market_data, dict) or not all(k in market_data for k in ['prices', 'volumes', 'dates']):
                logging.error(f"Invalid market data structure for symbol: {symbol}")
                return self._create_error_response(required_fields["timestamp"], f"Invalid market data structure for symbol: {symbol}")

        # Extract remaining optional fields with defaults
        risk_tolerance = data.get("risk_tolerance", 0.5)
        historical_decisions = data.get("historical_decisions", [])
        
        # Extract news analysis text from the news analysis dictionary
        news_data = required_fields["news_analysis"]
        latest_mi_summary = news_data.get("news_analysis", "") if isinstance(news_data, dict) else str(news_data)
        has_news = news_data.get("has_news", False) if isinstance(news_data, dict) else bool(latest_mi_summary.strip())

        logging.info(f"Starting analysis for {len(required_fields['symbols'])} symbols with {len(historical_decisions)} historical decisions")
        logging.info(f"News data available: {has_news}")

        # Continue with the rest of the analysis...
        symbols = data.get("symbols", [])
        timestamp = data.get("timestamp")
        risk_tolerance = data.get("risk_tolerance", 0.5)

        if not symbols or not market_data:
            logging.warning("Missing required data: symbols or current market data")
            return {
                "timestamp": timestamp,
                "error": "Insufficient input data for analysis",
                "market_intelligence": {"latest_summary": "", "past_summary": "", "sentiment": "NEUTRAL"},
                "low_level_reflection": {"reasonings": {}, "price_analysis": {}},
                "high_level_reflection": {"summary": "", "improvements": "", "trading_context": ""},
                "reflection_analysis": {"action": "HOLD", "reasoning": "Insufficient input data", "analysis": "", "confidence_score": 0.1}
            }

        # (3) Price Data Analysis - Move this before MI analysis to ensure data availability
        end_time = self._parse_timestamp(timestamp)
        start_time = end_time - timedelta(days=30)
        kline_data = {}
        price_data_available = False
        
        for symbol in symbols:
            data = self._fetch_kline_data(symbol, start_time, end_time)
            kline_data[symbol] = data
            if not data.startswith("Error") and not data.startswith("No price data"):
                price_data_available = True

        if not price_data_available:
            logging.warning("No valid price data available for any symbol")
            return {
                "timestamp": timestamp,
                "error": "No valid price data available",
                "market_intelligence": {"latest_summary": "", "past_summary": "", "sentiment": "NEUTRAL"},
                "low_level_reflection": {"reasonings": {}, "price_analysis": {}},
                "high_level_reflection": {"summary": "", "improvements": "", "trading_context": ""},
                "reflection_analysis": {"action": "HOLD", "reasoning": "No price data available", "analysis": "", "confidence_score": 0.1}
            }

        # (1) Market Intelligence Analysis with price context
        role = "You are a market intelligence analyst."
        content = f"Latest News Analysis for {symbols}:\n{latest_mi_summary}\n\n"
        content += "Recent Price Data:\n"
        for symbol, data in kline_data.items():
            content += f"{data}\n"
        content += "\nBased on both news and price data above, please provide a structured analysis in the following format:\n\n"
        content += "Summary: [Analyze both news and price movements, highlighting correlations and divergences]\n\n"
        content += "Sentiment: [State POSITIVE, NEGATIVE, or NEUTRAL with evidence from both news and price action]\n\n"
        content += "Short-Term Query: [Keywords for immediate market dynamics and catalysts]\n\n"
        content += "Medium-Term Query: [Keywords for emerging patterns and trends]\n\n"
        content += "Long-Term Query: [Keywords for fundamental shifts and cycles]"
        
        latest_mi_response = self._create_prompt(role, content)
        latest_mi_summary_parsed, latest_mi_queries, sentiment = self._parse_mi_response(latest_mi_response)

        # (2) Historical Market Intelligence Retrieval
        past_mi_items = self._retrieve_past_mi(latest_mi_queries)
        
        role = "You are a market intelligence analyst focusing on historical patterns."
        content = "Analyze the following historical market intelligence:\n"
        for item in past_mi_items:
            content += f"- {item['content']}\n"
        content += "\nProvide a comprehensive summary focusing on:\n"
        content += "1. Recurring patterns\n"
        content += "2. Historical precedents\n"
        content += "3. Market cycle indicators"
        
        past_mi_response = self._create_prompt(role, content)
        past_mi_summary = self._parse_past_mi_response(past_mi_response)

        # Store MI analysis in memory
        self.save_to_memory({
            "timestamp": timestamp,
            "latest_mi_summary": latest_mi_summary_parsed,
            "past_mi_summary": past_mi_summary,
            "sentiment": sentiment
        })
        
        # (4) Low-level Reflection (LLR) with enhanced context
        role = "You are a low-level reflection analyst."
        llr_content = "Analyze the following comprehensive market data:\n\n"
        llr_content += f"Market Intelligence:\n{latest_mi_summary_parsed}\n\n"
        llr_content += f"Historical Context:\n{past_mi_summary}\n\n"
        llr_content += "Technical Analysis:\n"
        for symbol, data in kline_data.items():
            llr_content += f"{data}\n"
        llr_content += f"\nCurrent Market State:\n{market_data}\n\n"
        llr_content += "Provide detailed analysis in the following format:\n\n"
        llr_content += "Short-Term Reasoning: [Analyze price action, volume patterns, and news catalysts]\n\n"
        llr_content += "Medium-Term Reasoning: [Identify trend formations, support/resistance levels, and pattern completions]\n\n"
        llr_content += "Long-Term Reasoning: [Evaluate market structure, cycle position, and fundamental shifts]\n\n"
        llr_content += "LLR Retrieval Query: [Key terms for similar technical setups and market conditions]"
        
        llr_response = self._create_prompt(role, llr_content)
        llr_reasonings, llr_query = self._parse_llr_response(llr_response)
        
        past_llr = self._retrieve_past_reflections(llr_query)

        # Store LLR analysis in memory
        self.save_to_memory({
            "timestamp": timestamp,
            "llr_reasonings": llr_reasonings,
            "price_data": kline_data
        })
        
        # (5) Trading Chart Analysis
        trading_chart_data = self._summarize_trading_chart_data(kline_data)

        # (6) High-level Reflection (HLR) with comprehensive data
        role = "You are a high-level reflection analyst."
        hlr_content = "Synthesize insights from the following comprehensive data:\n\n"
        hlr_content += f"Historical Performance:\n{historical_decisions}\n\n"
        hlr_content += f"Market Context:\n{market_data}\n\n"
        hlr_content += f"Price Analysis:\n{trading_chart_data}\n\n"
        hlr_content += f"Recent Analysis:\n"
        hlr_content += f"- Short-term: {llr_reasonings['short_term']}\n"
        hlr_content += f"- Medium-term: {llr_reasonings['medium_term']}\n"
        hlr_content += f"- Long-term: {llr_reasonings['long_term']}\n\n"
        hlr_content += "Provide strategic analysis in the following format:\n\n"
        hlr_content += "HLR Reasoning: [Evaluate strategy performance, success patterns, and failure points]\n\n"
        hlr_content += "HLR Improvement: [Suggest specific strategy adjustments based on historical results]\n\n"
        hlr_content += "HLR Summary: [Synthesize all insights into actionable trading strategy]\n\n"
        hlr_content += "HLR Retrieval Query: [Key terms for similar market conditions and strategy performance]"
        
        hlr_response = self._create_prompt(role, hlr_content)
        hlr_reasoning, hlr_improvement, hlr_summary, hlr_query = self._parse_hlr_response(hlr_response)
        
        past_hlr = self._retrieve_past_hlr(hlr_query)
        
        # Store HLR analysis in memory
        self.save_to_memory({
            "timestamp": timestamp,
            "hlr_summary": hlr_summary,
            "hlr_improvement": hlr_improvement,
            "trading_context": trading_chart_data
        })
        
        # (7) Final Decision Synthesis
        role = "You are a decision-maker with financial expertise."
        dm_content = "Based on the comprehensive market analysis below, provide a clear trading decision:\n\n"
        dm_content += f"Market Intelligence:\n"
        dm_content += f"- Latest Analysis: {latest_mi_summary_parsed}\n"
        dm_content += f"- Historical Context: {past_mi_summary}\n"
        dm_content += f"- Market Sentiment: {sentiment}\n\n"
        dm_content += f"Technical Analysis:\n"
        dm_content += f"- Short-term: {llr_reasonings['short_term']}\n"
        dm_content += f"- Medium-term: {llr_reasonings['medium_term']}\n"
        dm_content += f"- Long-term: {llr_reasonings['long_term']}\n\n"
        dm_content += f"Strategy Review:\n"
        dm_content += f"- Performance Analysis: {hlr_summary}\n"
        dm_content += f"- Suggested Improvements: {hlr_improvement}\n\n"
        dm_content += f"Risk Parameters:\n"
        dm_content += f"- Risk Tolerance: {risk_tolerance}\n"
        dm_content += f"- Historical Context: {len(historical_decisions)} past decisions analyzed\n"
        dm_content += f"- Market Coverage: {len(market_data)} symbols monitored\n\n"
        dm_content += "Provide your decision in the following format:\n\n"
        dm_content += "Action: [Specify BUY, SELL, or HOLD with clear conviction level]\n\n"
        dm_content += "Reasoning: [Provide detailed justification linking market intelligence, technical analysis, and strategy insights]\n\n"
        dm_content += "Analysis: [Detail key factors influencing the decision, potential risks, and expected outcomes]"
        
        dm_response = self._create_prompt(role, dm_content)
        decision, decision_reasoning, decision_analysis = self._parse_dm_response(dm_response)
        
        # Calculate confidence score based on consensus and analysis depth
        confidence_score = self._calculate_confidence_score(
            latest_mi_summary_parsed,
            llr_reasonings,
            hlr_summary,
            decision_analysis
        )
        
        final_result = {
            "timestamp": timestamp,
            "market_intelligence": {
                "latest_summary": latest_mi_summary_parsed,
                "past_summary": past_mi_summary,
                "sentiment": sentiment
            },
            "low_level_reflection": {
                "reasonings": llr_reasonings,
                "price_analysis": kline_data
            },
            "high_level_reflection": {
                "summary": hlr_summary,
                "improvements": hlr_improvement,
                "trading_context": trading_chart_data
            },
            "reflection_analysis": {
                "action": decision,
                "reasoning": decision_reasoning,
                "analysis": decision_analysis,
                "confidence_score": confidence_score
            }
        }
        
        # Store final analysis in memory
        self.save_to_memory(final_result)
        return final_result
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        try:
            dt = datetime.fromisoformat(timestamp_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.UTC)
            return dt
        except ValueError:
            return datetime.now(pytz.UTC)
    
    def _parse_mi_response(self, response: str) -> tuple:
        """Parse market intelligence response using flexible content-based extraction"""
        try:
            # Split response into paragraphs for analysis
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            
            # Initialize variables
            summary = []
            sentiment = "NEUTRAL"
            queries = {
                "short_term_query": [],
                "medium_term_query": [],
                "long_term_query": []
            }
            
            # Keywords for classification
            sentiment_keywords = {
                "POSITIVE": ["bullish", "positive", "optimistic", "upward", "growth", "strong", "increase"],
                "NEGATIVE": ["bearish", "negative", "pessimistic", "downward", "decline", "weak", "decrease"]
            }
            
            timeframe_keywords = {
                "short_term_query": ["immediate", "short-term", "current", "daily", "intraday"],
                "medium_term_query": ["medium-term", "weekly", "monthly", "intermediate"],
                "long_term_query": ["long-term", "yearly", "fundamental", "structural"]
            }
            
            # Analyze each paragraph
            for para in paragraphs:
                para_lower = para.lower()
                
                # Sentiment analysis
                positive_count = sum(1 for word in sentiment_keywords["POSITIVE"] if word in para_lower)
                negative_count = sum(1 for word in sentiment_keywords["NEGATIVE"] if word in para_lower)
                
                if positive_count > negative_count:
                    sentiment = "POSITIVE"
                elif negative_count > positive_count:
                    sentiment = "NEGATIVE"
                
                # Extract relevant keywords for queries
                for timeframe, keywords in timeframe_keywords.items():
                    if any(keyword in para_lower for keyword in keywords):
                        # Extract key phrases (3-word combinations)
                        words = para_lower.split()
                        for i in range(len(words)-2):
                            phrase = " ".join(words[i:i+3])
                            if any(keyword in phrase for keyword in keywords):
                                queries[timeframe].append(phrase)
                
                # Add paragraph to summary if it contains meaningful analysis
                if len(para.split()) >= 10:  # Minimum word count for meaningful content
                    summary.append(para)
            
            # Process collected data
            final_summary = "\n\n".join(summary) if summary else "Analysis pending due to insufficient content."
            
            # Convert query lists to strings
            final_queries = {}
            for timeframe, phrases in queries.items():
                if phrases:
                    # Take the most relevant phrase (first one found)
                    final_queries[timeframe] = phrases[0].replace(" ", "_")
                else:
                    # Default queries if none found
                    defaults = {
                        "short_term_query": "market_dynamics",
                        "medium_term_query": "market_trends",
                        "long_term_query": "market_structure"
                    }
                    final_queries[timeframe] = defaults[timeframe]
            
            return final_summary, final_queries, sentiment
            
        except Exception as e:
            logging.error(f"Error in flexible MI parsing: {str(e)}")
            return "Error processing market intelligence.", {
                "short_term_query": "market_dynamics",
                "medium_term_query": "market_trends",
                "long_term_query": "market_structure"
            }, "NEUTRAL"
    
    def _parse_llr_response(self, response: str) -> tuple:
        """Parse low-level reflection response using content-based extraction"""
        try:
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            
            # Initialize analysis categories
            analysis = {
                "short_term": [],
                "medium_term": [],
                "long_term": []
            }
            
            timeframe_indicators = {
                "short_term": ["immediate", "short-term", "intraday", "current", "today"],
                "medium_term": ["medium-term", "intermediate", "weekly", "monthly"],
                "long_term": ["long-term", "fundamental", "structural", "yearly"]
            }
            
            # Analyze each paragraph
            for para in paragraphs:
                para_lower = para.lower()
                
                # Classify paragraph by timeframe
                for timeframe, indicators in timeframe_indicators.items():
                    if any(indicator in para_lower for indicator in indicators):
                        analysis[timeframe].append(para)
                        break
            
            # Process collected analysis
            reasonings = {}
            for timeframe, content in analysis.items():
                if content:
                    reasonings[timeframe] = "\n".join(content)
                else:
                    reasonings[timeframe] = f"{timeframe.replace('_', ' ').title()} analysis pending."
            
            # Extract query terms based on content
            query_terms = set()
            for para in paragraphs:
                # Extract key technical terms
                technical_patterns = ["trend", "support", "resistance", "volume", "momentum", "volatility"]
                for pattern in technical_patterns:
                    if pattern in para.lower():
                        query_terms.add(pattern)
            
            query = "_".join(query_terms) if query_terms else "market_pattern_analysis"
            
            return reasonings, query
            
        except Exception as e:
            logging.error(f"Error in flexible LLR parsing: {str(e)}")
            return {
                "short_term": "Error processing short-term analysis.",
                "medium_term": "Error processing medium-term analysis.",
                "long_term": "Error processing long-term analysis."
            }, "market_analysis"
    
    def _parse_hlr_response(self, response: str) -> tuple:
        """Parse high-level reflection response using content-based extraction"""
        try:
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            
            # Initialize categories
            categories = {
                "reasoning": [],
                "improvement": [],
                "summary": []
            }
            
            # Keywords for classification
            category_keywords = {
                "reasoning": ["because", "therefore", "since", "as", "due to", "leads to", "results in"],
                "improvement": ["improve", "enhance", "optimize", "adjust", "modify", "update", "change"],
                "summary": ["overall", "in conclusion", "summary", "ultimately", "finally", "in summary"]
            }
            
            # Analyze each paragraph
            for para in paragraphs:
                para_lower = para.lower()
                
                # Classify paragraph by content
                max_matches = 0
                best_category = None
                
                for category, keywords in category_keywords.items():
                    matches = sum(1 for keyword in keywords if keyword in para_lower)
                    if matches > max_matches:
                        max_matches = matches
                        best_category = category
                
                if best_category:
                    categories[best_category].append(para)
                else:
                    # If no clear category, add to summary
                    categories["summary"].append(para)
            
            # Process collected content
            reasoning = "\n".join(categories["reasoning"]) if categories["reasoning"] else "Strategy analysis pending."
            improvement = "\n".join(categories["improvement"]) if categories["improvement"] else "Improvement suggestions pending."
            summary = "\n".join(categories["summary"]) if categories["summary"] else "Analysis summary pending."
            
            # Generate query based on content
            query_terms = set()
            for para in paragraphs:
                strategy_terms = ["strategy", "performance", "risk", "return", "allocation", "position"]
                for term in strategy_terms:
                    if term in para.lower():
                        query_terms.add(term)
            
            query = "_".join(query_terms) if query_terms else "trading_strategy_analysis"
            
            return reasoning, improvement, summary, query
            
        except Exception as e:
            logging.error(f"Error in flexible HLR parsing: {str(e)}")
            return (
                "Error processing strategy analysis.",
                "Error generating improvements.",
                "Error creating summary.",
                "strategy_analysis"
            )
    
    def _parse_dm_response(self, response: str) -> tuple:
        """Parse decision maker response using content-based extraction"""
        try:
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            
            # Initialize variables
            action = "HOLD"  # Default action
            reasoning = []
            analysis = []
            
            # Action keywords
            action_keywords = {
                "BUY": ["buy", "long", "purchase", "accumulate", "enter long"],
                "SELL": ["sell", "short", "reduce", "exit", "enter short"],
                "HOLD": ["hold", "maintain", "keep", "stay", "neutral"]
            }
            
            # Analysis keywords
            analysis_keywords = ["because", "due to", "based on", "given", "considering"]
            
            # Process each paragraph
            for para in paragraphs:
                para_lower = para.lower()
                
                # Determine action
                for act, keywords in action_keywords.items():
                    if any(keyword in para_lower for keyword in keywords):
                        action = act
                        break
                
                # Classify content
                if any(keyword in para_lower for keyword in analysis_keywords):
                    if len(para.split()) > 20:  # Detailed analysis
                        analysis.append(para)
                    else:  # Brief reasoning
                        reasoning.append(para)
                else:
                    reasoning.append(para)
            
            final_reasoning = "\n".join(reasoning) if reasoning else "Reasoning pending due to insufficient data."
            final_analysis = "\n".join(analysis) if analysis else "Detailed analysis pending."
            
            return action, final_reasoning, final_analysis
            
        except Exception as e:
            logging.error(f"Error in flexible DM parsing: {str(e)}")
            return "HOLD", "Error processing decision.", "Error in analysis."
    
    def _retrieve_past_mi(self, queries: Dict[str, str]) -> List[Dict[str, str]]:
        """Retrieve past market intelligence based on query strings"""
        past_mi = []
        try:
            for timeframe, query in queries.items():
                # Ensure query is a string
                query_str = str(query).lower()
                
                # Find matching items in memory
                matching_items = [
                    item for item in self.memory.get('market_intelligence', [])
                    if isinstance(item.get('content', ''), (str, bytes)) and 
                    query_str in str(item.get('content', '')).lower()
                ]
                
                if matching_items:
                    # Get last 3 matching items
                    past_mi.extend(matching_items[-3:])
                else:
                    past_mi.append({
                        "content": f"Historical {timeframe} data: No significant patterns found."
                    })
                    
        except Exception as e:
            logging.error(f"Error retrieving past MI: {str(e)}")
            past_mi.append({
                "content": "Error retrieving historical market intelligence data."
            })
            
        return past_mi
    
    def _parse_past_mi_response(self, response: str) -> str:
        return self._extract_line(response, "Overall Analysis:", default="No historical patterns identified.")
    
    def _fetch_kline_data(self, symbol: str, start: datetime, end: datetime) -> str:
        """Fetch and analyze price data with enhanced technical analysis"""
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if df.empty:
                return f"No price data available for {symbol}"
            
            # Basic price analysis
            recent_close = df['Close'].iloc[-1]
            old_close = df['Close'].iloc[0]
            change = (recent_close - old_close) / old_close * 100
            
            # Technical indicators
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            trend = "UPTREND" if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] else "DOWNTREND"
            
            # Volatility analysis
            volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
            recent_volatility = df['Close'].pct_change().tail(5).std() * np.sqrt(252) * 100
            vol_trend = "INCREASING" if recent_volatility > volatility else "DECREASING"
            
            # Volume analysis
            volume_change = (df['Volume'].iloc[-1] / df['Volume'].iloc[0] - 1) * 100
            avg_volume = df['Volume'].mean()
            vol_strength = "HIGH" if df['Volume'].iloc[-1] > avg_volume * 1.5 else "NORMAL"
            
            return (f"{symbol} Technical Analysis:\n"
                   f"- Price: {old_close:.2f} → {recent_close:.2f} ({change:+.2f}%)\n"
                   f"- Trend: {trend} (SMA20 vs SMA50)\n"
                   f"- Volatility: {volatility:.2f}% ({vol_trend})\n"
                   f"- Volume: {volume_change:+.2f}% change, {vol_strength} relative strength\n"
                   f"- Support Levels: {df['Low'].min():.2f}, {df['Low'].nlargest(2).iloc[-1]:.2f}\n"
                   f"- Resistance Levels: {df['High'].nsmallest(2).iloc[-1]:.2f}, {df['High'].max():.2f}")
        except Exception as e:
            return f"Error fetching data for {symbol}: {str(e)}"
    
    def _retrieve_past_reflections(self, query: str) -> List[Dict[str, str]]:
        # Retrieve from memory or return placeholder
        matching_reflections = [
            item for item in self.memory.get('reflections', [])
            if query.lower() in str(item).lower()
        ]
        if matching_reflections:
            return matching_reflections[-3:]  # Get last 3 matching reflections
        return [{"content": "No matching historical reflections found."}]
    
    def _summarize_trading_chart_data(self, kline_data: Dict[str, str]) -> str:
        summary_parts = []
        for symbol, data in kline_data.items():
            summary_parts.append(f"Symbol: {symbol}\n{data}")
        return "\n\n".join(summary_parts)
    
    def _retrieve_past_hlr(self, query: str) -> List[Dict[str, str]]:
        # Retrieve from memory or return placeholder
        matching_hlr = [
            item for item in self.memory.get('high_level_reflections', [])
            if query.lower() in str(item).lower()
        ]
        if matching_hlr:
            return matching_hlr[-3:]  # Get last 3 matching HLR items
        return [{"content": "No matching high-level reflections found."}]
    
    def _calculate_confidence_score(self, mi_summary: str, llr: Dict[str, str], 
                                 hlr_summary: str, decision_analysis: str) -> float:
        """Calculate confidence score based on analysis depth and consensus"""
        score = 0.5  # Base confidence
        
        # Check for comprehensive analysis
        if len(mi_summary) > 100: score += 0.1
        if all(len(r) > 50 for r in llr.values()): score += 0.1
        if len(hlr_summary) > 100: score += 0.1
        if len(decision_analysis) > 100: score += 0.1
        
        # Check for consensus indicators
        consensus_terms = ['clearly', 'strongly', 'definitely', 'consistently']
        uncertainty_terms = ['possibly', 'might', 'unclear', 'uncertain']
        
        text_to_analyze = f"{mi_summary} {' '.join(llr.values())} {hlr_summary} {decision_analysis}".lower()
        
        consensus_count = sum(term in text_to_analyze for term in consensus_terms)
        uncertainty_count = sum(term in text_to_analyze for term in uncertainty_terms)
        
        score += 0.02 * consensus_count
        score -= 0.02 * uncertainty_count
        
        return max(0.1, min(0.9, score))
    
    def _extract_line(self, text: str, prefix: str, default: str = "") -> str:
        """Extract line starting with prefix from text"""
        for line in text.splitlines():
            if line.strip().startswith(prefix):
                return line.split(prefix, 1)[1].strip()
        return default
    
    def save_to_memory(self, data: Dict[str, Any]) -> None:
        """Save data to agent's memory with timestamp"""
        timestamp = data.get('timestamp', datetime.now().isoformat())
        for key, value in data.items():
            if key != 'timestamp':
                self.memory[key].append({
                    'timestamp': timestamp,
                    'content': value
                })
    
    def _create_error_response(self, timestamp: str, error_message: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            "timestamp": timestamp,
            "error": error_message,
            "market_intelligence": {
                "latest_summary": "",
                "past_summary": "",
                "sentiment": "NEUTRAL"
            },
            "low_level_reflection": {
                "reasonings": {},
                "price_analysis": {}
            },
            "high_level_reflection": {
                "summary": "",
                "improvements": "",
                "trading_context": ""
            },
            "reflection_analysis": {
                "action": "HOLD",
                "reasoning": f"Analysis failed: {error_message}",
                "analysis": "",
                "confidence_score": 0.1
            }
        }
    
    def _validate_market_data(self, market_data: Dict[str, Any], symbol: str) -> bool:
        """Validate the structure and content of market data for a symbol"""
        try:
            if not isinstance(market_data, dict):
                logging.error(f"Market data for {symbol} is not a dictionary")
                return False
                
            required_fields = ['prices', 'volumes', 'dates']
            for field in required_fields:
                if field not in market_data:
                    logging.error(f"Missing {field} in market data for {symbol}")
                    return False
                    
            if not isinstance(market_data['prices'], list) or not market_data['prices']:
                logging.error(f"Invalid or empty prices data for {symbol}")
                return False
                
            if not isinstance(market_data['volumes'], list) or not market_data['volumes']:
                logging.error(f"Invalid or empty volumes data for {symbol}")
                return False
                
            if not isinstance(market_data['dates'], list) or not market_data['dates']:
                logging.error(f"Invalid or empty dates data for {symbol}")
                return False
                
            if not (len(market_data['prices']) == len(market_data['volumes']) == len(market_data['dates'])):
                logging.error(f"Mismatched data lengths for {symbol}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating market data for {symbol}: {str(e)}")
            return False