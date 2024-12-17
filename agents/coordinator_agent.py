from typing import Dict, Any, List
from agents import BaseAgent
from agents.news_agent import NewsAgent
from agents.reflection_agent import ReflectionAgent
from agents.debate_agent import DebateAgent
from config import AGENT_SETTINGS

class CoordinatorAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.news_agent = NewsAgent(AGENT_SETTINGS["news_agent"])
        self.reflection_agent = ReflectionAgent(AGENT_SETTINGS["reflection_agent"])
        self.debate_agent = DebateAgent(AGENT_SETTINGS["debate_agent"])
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate analysis from all agents and make final decision
        
        Args:
            data: Dictionary containing market data and trading context
            
        Returns:
            Dictionary containing final analysis and trading decision
        """
        enabled_agents = data.get("enabled_agents", {
            "news_agent": True,
            "reflection_agent": True,
            "debate_agent": True
        })
        
        analyses = {}
        
        # Collect insights from enabled agents
        if enabled_agents.get("news_agent", True):
            news_analysis = self.news_agent.analyze({
                "symbols": data.get("symbols", []),
                "timestamp": data.get("timestamp")
            })
            analyses["news"] = news_analysis
            print(f"News Analysis: {news_analysis}")
        else:
            analyses["news"] = {
                "news_analysis": "News Agent is disabled",
                "analyzed_symbols": data.get("symbols", []),
                "timestamp": data.get("timestamp")
            }
        
        if enabled_agents.get("reflection_agent", True):
            reflection_analysis = self.reflection_agent.analyze({
                "symbols": data.get("symbols", []),
                "news_analysis": analyses["news"],
                "risk_tolerance": data.get("risk_tolerance", 0.5),
                "historical_decisions": data.get("historical_decisions", []),
                "market_data": data.get("market_data", {}),
                "timestamp": data.get("timestamp")
            })
            analyses["reflection"] = reflection_analysis
            print(f"Reflection Analysis: {reflection_analysis}")
        else:
            analyses["reflection"] = {
                "reflection_analysis": "Reflection Agent is disabled",
                "timestamp": data.get("timestamp"),
                "patterns_identified": [{"pattern_type": "disabled", "success_rate": 0.5}]
            }
        
        if enabled_agents.get("debate_agent", True):
            debate_analysis = self.debate_agent.analyze({
                "market_data": data.get("market_data", {}),
                "proposed_action": data.get("proposed_action", {}),
                "timestamp": data.get("timestamp")
            })
            analyses["debate"] = debate_analysis
            print(f"Debate Analysis: {debate_analysis}")
        else:
            analyses["debate"] = {
                "debate_analysis": "Debate Agent is disabled",
                "timestamp": data.get("timestamp"),
                "confidence_score": 0.5
            }
        
        # Synthesize all analyses
        final_decision = self._synthesize_analyses(
            analyses["news"],
            analyses["reflection"],
            analyses["debate"],
            data
        )

        print(f"Final decision: {final_decision}")
        
        analysis_result = {
            "final_decision": final_decision,
            "component_analyses": analyses,
            "timestamp": data.get("timestamp"),
            "confidence_score": self._calculate_overall_confidence(
                analyses["news"],
                analyses["reflection"],
                analyses["debate"],
                enabled_agents
            ),
        }
        
        self.save_to_memory(analysis_result)
        return analysis_result
    
    def _synthesize_analyses(
        self,
        news_analysis: Dict[str, Any] | None,
        reflection_analysis: Dict[str, Any] | None,
        debate_analysis: Dict[str, Any] | None,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize analyses from all agents into final decision"""
        # Handle None values with default empty dictionaries
        news_analysis = news_analysis or {}
        reflection_analysis = reflection_analysis or {}
        debate_analysis = debate_analysis or {}

        role = """You are a master trading strategist responsible for making the final 
        trading decision based on multiple perspectives and analyses. For each symbol,
        you must provide a clear BULLISH or BEARISH signal.
        
        Your response MUST end with a section formatted EXACTLY like this:
        
        SIGNALS:
        SYMBOL: <symbol1> | SIGNAL: BULLISH
        SYMBOL: <symbol2> | SIGNAL: BEARISH
        
        Use ONLY 'BULLISH' or 'BEARISH' as signals - no other variations are allowed.
        The format must be exact with the pipe character | separating the symbol and signal."""
        
        content = f"""
        Synthesize the following analyses into a final trading decision:
        
        News Analysis:
        {news_analysis.get('news_analysis', 'No news analysis available')}
        
        Historical Reflection:
        {reflection_analysis.get('reflection_analysis', 'No reflection analysis available')}
        
        Debate Conclusion:
        {debate_analysis.get('debate_analysis', 'No debate analysis available')}
        
        Market Context:
        {context.get('market_data', {})}
        
        Provide a final decision addressing:
        1. Decision Summary
        2. Key Factors Considered
        3. Risk Management Strategy
        4. Implementation Plan
        5. Monitoring Criteria
        
        Then, provide clear BULLISH or BEARISH signals for each symbol:
        {', '.join(context.get('symbols', []))}
        
        Your response MUST end with the SIGNALS section formatted exactly as specified above.
        Each symbol MUST have either BULLISH or BEARISH signal - no other variations allowed.
        Follow this exact format for each symbol:
        SYMBOL: <symbol> | SIGNAL: <BULLISH or BEARISH>
        """
        
        response = self._create_prompt(role, content)
        
        # Extract boolean signals for each symbol
        symbols = context.get('symbols', [])
        symbol_signals = {}
        
        # Find the SIGNALS section
        if "SIGNALS:" in response:
            signals_section = response.split("SIGNALS:")[1].strip()
            signal_lines = [line.strip() for line in signals_section.split('\n') if line.strip()]
            
            for line in signal_lines:
                if "SYMBOL:" in line and "SIGNAL:" in line and "|" in line:
                    try:
                        # Split by | to separate symbol and signal parts
                        parts = line.split("|")
                        if len(parts) != 2:
                            continue
                            
                        symbol_part = parts[0].split("SYMBOL:")[1].strip()
                        signal_part = parts[1].split("SIGNAL:")[1].strip().upper()
                        
                        # Only accept exact BULLISH or BEARISH signals
                        if signal_part not in ["BULLISH", "BEARISH"]:
                            continue
                            
                        symbol = symbol_part.strip()
                        is_bullish = signal_part == "BULLISH"
                        
                        if symbol in symbols:
                            symbol_signals[symbol] = is_bullish
                            print(f"Parsed signal for {symbol}: {'BULLISH' if is_bullish else 'BEARISH'}")
                    except Exception as e:
                        print(f"Error parsing signal line: {line}, Error: {str(e)}")
                        continue
        
        # For any missing symbols, use sentiment analysis
        for symbol in symbols:
            if symbol not in symbol_signals:
                symbol_section = self._find_symbol_section(response, symbol)
                if symbol_section:
                    is_bullish = self._analyze_sentiment(symbol_section)
                    symbol_signals[symbol] = is_bullish
                    print(f"Used sentiment analysis for {symbol}: {'BULLISH' if is_bullish else 'BEARISH'}")
                else:
                    # Default to bearish only if we can't find any mention
                    print(f"Warning: No signal found for {symbol}, defaulting to bearish")
                    symbol_signals[symbol] = False
        
        return {
            "decision": response,
            "timestamp": context.get("timestamp"),
            "market_context": context.get("market_data", {}),
            "symbol_signals": symbol_signals
        }
    
    def _find_symbol_section(self, text: str, symbol: str) -> str:
        """Find the section of text discussing a specific symbol"""
        paragraphs = text.split('\n\n')
        relevant_text = []
        
        for para in paragraphs:
            if symbol in para:
                relevant_text.append(para)
        
        return '\n'.join(relevant_text)
    
    def _analyze_sentiment(self, text: str) -> bool:
        """Analyze text sentiment to determine if it's bullish or bearish"""
        # Bullish indicators
        bullish_phrases = [
            "bullish", "positive outlook", "upward trend", "growth potential",
            "buy signal", "upside potential", "strong performance", "outperform",
            "recommend buy", "price target up", "upgrade", "momentum", "strong buy"
        ]
        
        # Bearish indicators
        bearish_phrases = [
            "bearish", "negative outlook", "downward trend", "decline",
            "sell signal", "downside risk", "weak performance", "underperform",
            "recommend sell", "price target down", "downgrade", "sell", "strong sell"
        ]
        
        # Count occurrences
        bullish_count = sum(1 for phrase in bullish_phrases if phrase.lower() in text.lower())
        bearish_count = sum(1 for phrase in bearish_phrases if phrase.lower() in text.lower())
        
        # Consider context modifiers
        if "not bullish" in text.lower() or "no longer bullish" in text.lower():
            bullish_count -= 1
        if "not bearish" in text.lower() or "no longer bearish" in text.lower():
            bearish_count -= 1
        
        # Return sentiment
        if bullish_count > bearish_count:
            return True
        return False
    
    def _calculate_overall_confidence(
        self,
        news_analysis: Dict[str, Any],
        reflection_analysis: Dict[str, Any],
        debate_analysis: Dict[str, Any],
        enabled_agents: Dict[str, bool]
    ) -> float:
        """Calculate overall confidence score based on enabled analyses"""
        confidence_scores = []
        total_weight = 0
        
        # Add debate confidence if enabled
        if enabled_agents.get("debate_agent", True):
            confidence_scores.append(debate_analysis.get("confidence_score", 0) * 0.4)
            total_weight += 0.5
        
        # Add reflection confidence if enabled
        if enabled_agents.get("reflection_agent", True):
            reflection_score = reflection_analysis.get("reflection_analysis", {}).get("confidence_score", 0) * 0.35
            confidence_scores.append(reflection_score)
            total_weight += 0.3
        
        # Add news confidence if enabled
        if enabled_agents.get("news_agent", True):
            confidence_scores.append(0.25)  # Base confidence from news analysis
            total_weight += 0.2
        
        # Calculate weighted average, ensuring we don't divide by zero
        if total_weight > 0:
            return sum(confidence_scores) * (1 / total_weight)
        return 0.5  # Default confidence if all agents are disabled
