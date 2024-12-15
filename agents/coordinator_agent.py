from typing import Dict, Any, List
from . import BaseAgent
from .news_agent import NewsAgent
from .reflection_agent import ReflectionAgent
from .debate_agent import DebateAgent
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
        # Collect insights from all agents
        news_analysis = self.news_agent.analyze({
            "symbols": data.get("symbols", []),
            "timestamp": data.get("timestamp")
        })
        
        reflection_analysis = self.reflection_agent.analyze({
            "historical_decisions": data.get("historical_decisions", []),
            "current_market": data.get("market_data", {}),
            "timestamp": data.get("timestamp")
        })
        
        debate_analysis = self.debate_agent.analyze({
            "market_data": data.get("market_data", {}),
            "proposed_action": data.get("proposed_action", {}),
            "timestamp": data.get("timestamp")
        })
        
        # Synthesize all analyses
        final_decision = self._synthesize_analyses(
            news_analysis,
            reflection_analysis,
            debate_analysis,
            data
        )
        
        analysis_result = {
            "final_decision": final_decision,
            "component_analyses": {
                "news": news_analysis,
                "reflection": reflection_analysis,
                "debate": debate_analysis
            },
            "timestamp": data.get("timestamp"),
            "confidence_score": self._calculate_overall_confidence(
                news_analysis,
                reflection_analysis,
                debate_analysis
            )
        }
        
        self.save_to_memory(analysis_result)
        return analysis_result
    
    def _synthesize_analyses(
        self,
        news_analysis: Dict[str, Any],
        reflection_analysis: Dict[str, Any],
        debate_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize analyses from all agents into final decision"""
        role = """You are a master trading strategist responsible for making the final 
        trading decision based on multiple perspectives and analyses."""
        
        content = f"""
        Synthesize the following analyses into a final trading decision:
        
        News Analysis:
        {news_analysis.get('news_analysis', '')}
        
        Historical Reflection:
        {reflection_analysis.get('reflection_analysis', '')}
        
        Debate Conclusion:
        {debate_analysis.get('debate_analysis', '')}
        
        Market Context:
        {context.get('market_data', {})}
        
        Provide a final decision addressing:
        1. Decision Summary
        2. Key Factors Considered
        3. Risk Management Strategy
        4. Implementation Plan
        5. Monitoring Criteria
        """
        
        response = self._create_prompt(role, content)
        
        return {
            "decision": response,
            "timestamp": context.get("timestamp"),
            "market_context": context.get("market_data", {})
        }
    
    def _calculate_overall_confidence(
        self,
        news_analysis: Dict[str, Any],
        reflection_analysis: Dict[str, Any],
        debate_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score based on all analyses"""
        # This would implement more sophisticated confidence calculation
        # For now, use a weighted average of component confidences
        confidence_scores = [
            debate_analysis.get("confidence_score", 0) * 0.4,  # Debate weighted highest
            reflection_analysis.get("patterns_identified", [{}])[0].get("success_rate", 0) * 0.35,
            0.25  # Base confidence from news analysis
        ]
        
        return sum(confidence_scores) 