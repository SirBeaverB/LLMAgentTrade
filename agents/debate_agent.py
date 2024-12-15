from typing import Dict, Any, List
from . import BaseAgent

class DebateAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.debate_rounds = 3
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct an internal debate about trading decisions
        
        Args:
            data: Dictionary containing market data and proposed actions
            
        Returns:
            Dictionary containing debate conclusions and recommendations
        """
        market_data = data.get("market_data", {})
        proposed_action = data.get("proposed_action", {})
        
        debate_results = self._conduct_debate(market_data, proposed_action)
        
        final_analysis = self._synthesize_debate(debate_results)
        
        analysis_result = {
            "debate_analysis": final_analysis,
            "debate_rounds": debate_results,
            "timestamp": data.get("timestamp"),
            "confidence_score": self._calculate_confidence(debate_results)
        }
        
        self.save_to_memory(analysis_result)
        return analysis_result
    
    def _conduct_debate(self, market_data: Dict[str, Any], proposed_action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Conduct multiple rounds of debate"""
        debate_rounds = []
        
        for round_num in range(self.debate_rounds):
            # Alternate between bull and bear perspectives
            is_bull = round_num % 2 == 0
            
            role = f"""You are a {'bullish' if is_bull else 'bearish'} market analyst in a 
            debate about a trading decision. Your goal is to present strong arguments 
            {'supporting' if is_bull else 'against'} the proposed action, considering all 
            available market data."""
            
            content = f"""
            Round {round_num + 1} of debate:
            
            Market Data:
            {market_data}
            
            Proposed Action:
            {proposed_action}
            
            Previous Arguments:
            {self._format_previous_rounds(debate_rounds)}
            
            Present your {'bullish' if is_bull else 'bearish'} argument, addressing:
            1. Market Conditions
            2. Technical Factors
            3. Risk Assessment
            4. Potential Outcomes
            """
            
            response = self._create_prompt(role, content)
            
            debate_rounds.append({
                "round": round_num + 1,
                "perspective": "bull" if is_bull else "bear",
                "arguments": response
            })
        
        return debate_rounds
    
    def _synthesize_debate(self, debate_rounds: List[Dict[str, Any]]) -> str:
        """Synthesize the debate rounds into a final analysis"""
        role = """You are a neutral trading analyst tasked with synthesizing multiple 
        perspectives from a debate into a balanced final analysis."""
        
        content = f"""
        Synthesize the following debate rounds into a final analysis:
        
        Debate History:
        {self._format_previous_rounds(debate_rounds)}
        
        Provide a balanced analysis addressing:
        1. Key Points of Agreement
        2. Major Points of Contention
        3. Risk-Reward Assessment
        4. Final Recommendation
        """
        
        return self._create_prompt(role, content)
    
    def _format_previous_rounds(self, debate_rounds: List[Dict[str, Any]]) -> str:
        """Format previous debate rounds for context"""
        if not debate_rounds:
            return "No previous arguments."
        
        formatted_rounds = []
        for round_data in debate_rounds:
            formatted_rounds.append(
                f"Round {round_data['round']} ({round_data['perspective'].upper()}):\n"
                f"{round_data['arguments']}\n"
            )
        
        return "\n".join(formatted_rounds)
    
    def _calculate_confidence(self, debate_rounds: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on debate strength"""
        # This would implement more sophisticated confidence calculation
        # For now, return a placeholder
        return 0.75 