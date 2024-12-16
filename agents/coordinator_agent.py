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
                "historical_decisions": data.get("historical_decisions", []),
                "current_market": data.get("market_data", {}),
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
        you must provide a clear boolean signal (True for bullish/positive, False for bearish/negative).
        
        Your response MUST include a section at the end formatted exactly like this:
        
        SIGNALS:
        SYMBOL: <symbol1> | SIGNAL: <BULLISH/BEARISH>
        SYMBOL: <symbol2> | SIGNAL: <BULLISH/BEARISH>
        ...etc
        
        Use BULLISH for positive outlook and BEARISH for negative outlook."""
        
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
        
        Then, provide clear signals for each symbol:
        {', '.join(context.get('symbols', []))}
        
        Your response MUST end with the signals section formatted exactly as specified above.
        """
        
        response = self._create_prompt(role, content)
        
        # Extract boolean signals for each symbol
        symbols = context.get('symbols', [])
        symbol_signals = {}
        
        # Find the SIGNALS section
        if "SIGNALS:" in response:
            signals_section = response.split("SIGNALS:")[1].strip()
            signal_lines = signals_section.split('\n')
            
            for line in signal_lines:
                if "SYMBOL:" in line and "SIGNAL:" in line:
                    # Parse each signal line
                    try:
                        symbol_part = line.split("SIGNAL:")[0].split("SYMBOL:")[1].strip()
                        signal_part = line.split("SIGNAL:")[1].strip()
                        
                        # Extract just the symbol (remove any '|' or other characters)
                        symbol = symbol_part.replace("|", "").strip()
                        
                        # Convert signal to boolean
                        is_bullish = signal_part.upper().strip() == "BULLISH"
                        
                        if symbol in symbols:
                            symbol_signals[symbol] = is_bullish
                    except Exception as e:
                        continue
        
        # If any symbols are missing from the parsed signals, analyze the full response
        for symbol in symbols:
            if symbol not in symbol_signals:
                # Look for explicit mentions of the symbol with sentiment
                symbol_section = self._find_symbol_section(response, symbol)
                if symbol_section:
                    # Analyze the section for sentiment
                    is_bullish = self._analyze_sentiment(symbol_section)
                    symbol_signals[symbol] = is_bullish
                else:
                    # Default to neutral/bearish if we can't find clear sentiment
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
            total_weight += 0.4
        
        # Add reflection confidence if enabled
        if enabled_agents.get("reflection_agent", True):
            reflection_score = reflection_analysis.get("patterns_identified", [{}])[0].get("success_rate", 0) * 0.35
            confidence_scores.append(reflection_score)
            total_weight += 0.35
        
        # Add news confidence if enabled
        if enabled_agents.get("news_agent", True):
            confidence_scores.append(0.25)  # Base confidence from news analysis
            total_weight += 0.25
        
        # Calculate weighted average, ensuring we don't divide by zero
        if total_weight > 0:
            return sum(confidence_scores) * (1 / total_weight)
        return 0.5  # Default confidence if all agents are disabled