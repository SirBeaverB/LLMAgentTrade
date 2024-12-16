from typing import Dict, Any, List
from . import BaseAgent

class DebateAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.debate_rounds = 3
        self.roles = [
                {
                    "name": "fundamental",
                    "description": """You are a Fundamental Analyst focusing on macroeconomic indicators,
                                    company financials, sector trends, and other fundamental factors that
                                    influence the asset's intrinsic value. 
                                    Your goal is to provide arguments about the proposed action from a fundamental perspective."""
                },
                {
                    "name": "technical",
                    "description": """You are a Technical Analyst focusing on price trends, chart patterns,
                                    technical indicators, and volume analysis.
                                    Your goal is to provide arguments about the proposed action from a technical perspective."""
                },
                {
                    "name": "risk",
                    "description": """You are a Risk Analyst focusing on potential risks, uncertainties, 
                                    volatility, regulatory changes, and market sentiment shifts.
                                    Your goal is to provide arguments about the proposed action from a risk management perspective."""
                }
            ]
    
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
            "confidence_score": self._calculate_confidence(debate_results, market_data)
        }
        
        self.save_to_memory(analysis_result)
        return analysis_result
    
    def _conduct_debate(self, market_data: Dict[str, Any], proposed_action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Conduct multiple rounds of debate"""
        debate_rounds = []
        
        for round_num in range(self.debate_rounds):
            # Alternate between bull and bear perspectives
            for role_info in self.roles:
                role = role_info["description"]
                perspective_name = role_info["name"]
                
                content = f"""
                Round {round_num + 1} of debate ({perspective_name.upper()} perspective):

                Market Data:
                {market_data}

                Proposed Action:
                {proposed_action}

                Previous Arguments:
                {self._format_previous_rounds(debate_rounds)}

                Present your {perspective_name} argument, addressing:
                1. Market Conditions
                2. Technical Factors
                3. Risk Assessment
                4. Potential Outcomes
                """

                response = self._create_prompt(role, content)

                debate_rounds.append({
                    "round": round_num + 1,
                    "perspective": perspective_name,
                    "arguments": response
                })
        
        return debate_rounds
    
    def _synthesize_debate(self, debate_rounds: List[Dict[str, Any]]) -> str:
        """Synthesize the debate rounds into a final analysis"""
        role = """You are a senior market strategist tasked with synthesizing insights 
                    from three specialized analysts (fundamental, technical, and risk). 
                    Combine their perspectives into a balanced and actionable final analysis."""
        
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
    
    def _calculate_confidence(self, debate_rounds: List[Dict[str, Any]], market_data: Dict[str, Any]) -> float:
        """
        Calculate the confidence score based on debate results from three roles (fundamental, technical, risk)
        and market data.

        Approach:
        1. Start with a base confidence value.
        2. Adjust confidence based on market trend.
        3. Extract arguments from fundamental, technical, and risk perspectives.
        4. Identify agreements and disagreements by comparing pairs of roles:
        (fundamental vs technical), (fundamental vs risk), and (technical vs risk).
        5. Increase confidence for agreements, decrease for disagreements.
        6. Check for technical consistency signals among these roles.
        7. Ensure confidence stays within [0.0, 1.0].
        """

        confidence = 0.5
        trend = market_data.get("trend", "flat")

        # Adjust confidence based on market trend
        if trend == "up":
            confidence += 0.1
        elif trend == "down":
            confidence -= 0.05
        else:
            confidence -= 0.02

        # Extract arguments for each role
        fundamental_args = [r['arguments'] for r in debate_rounds if r['perspective'] == 'fundamental']
        technical_args   = [r['arguments'] for r in debate_rounds if r['perspective'] == 'technical']
        risk_args        = [r['arguments'] for r in debate_rounds if r['perspective'] == 'risk']

        # Combine arguments
        all_fundamental_text = " ".join(fundamental_args).lower()
        all_technical_text = " ".join(technical_args).lower()
        all_risk_text = " ".join(risk_args).lower()

        agreements = 0
        disagreements = 0

        # A helper function to check agreements/disagreements between two role texts
        def check_pairwise(text_a: str, text_b: str):
            local_agreements = 0
            local_disagreements = 0

            # Example: "mild increase" considered an agreement if both mention it
            if "mild increase" in text_a and "mild increase" in text_b:
                local_agreements += 1

            # Risk assessment keyword checks
            if "low risk" in text_a and "low risk" in text_b:
                local_agreements += 1
            elif ("low risk" in text_a and "high risk" in text_b) or ("high risk" in text_a and "low risk" in text_b):
                local_disagreements += 1

            # Trend disagreement: one sees upward momentum, the other downward pressure
            if "downward pressure" in text_a and "upward momentum" in text_b:
                local_disagreements += 1
            elif "downward pressure" in text_b and "upward momentum" in text_a:
                local_disagreements += 1

            return local_agreements, local_disagreements

        # Compare fundamental vs technical
        a, d = check_pairwise(all_fundamental_text, all_technical_text)
        agreements += a
        disagreements += d

        # Compare fundamental vs risk
        a, d = check_pairwise(all_fundamental_text, all_risk_text)
        agreements += a
        disagreements += d

        # Compare technical vs risk
        a, d = check_pairwise(all_technical_text, all_risk_text)
        agreements += a
        disagreements += d

        confidence += (agreements * 0.05)
        confidence -= (disagreements * 0.05)

        # Check if at least two perspectives mention the same technical phrase.
        if ("moving average crossing above price" in all_fundamental_text and
            "moving average crossing above price" in all_technical_text) or \
        ("moving average crossing above price" in all_fundamental_text and
            "moving average crossing above price" in all_risk_text) or \
        ("moving average crossing above price" in all_technical_text and
            "moving average crossing above price" in all_risk_text):
            confidence += 0.03

        confidence = max(0.0, min(1.0, confidence))

        return confidence