from typing import Dict, Any, List
import pandas as pd
from agents import BaseAgent

class ReflectionAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.historical_decisions = []
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze historical decisions and current market patterns
        
        Args:
            data: Dictionary containing historical decisions and current market data
            
        Returns:
            Dictionary containing reflection analysis and recommendations
        """
        historical_data = self._prepare_historical_data(data)
        current_market = data.get("current_market", {})
        
        role = """You are a reflective trading analyst who learns from past decisions and 
        market patterns. Analyze historical trading decisions and current market conditions 
        to provide insights and recommendations."""
        
        content = f"""Analyze the following historical trading decisions and current market conditions:
        
        Historical Decisions:
        {historical_data}
        
        Current Market Conditions:
        {current_market}
        
        Provide analysis in the following format:
        1. Pattern Recognition
        2. Historical Performance Analysis
        3. Lessons Learned
        4. Recommended Adjustments
        5. Risk Assessment
        """
        
        response = self._create_prompt(role, content)
        
        analysis_result = {
            "reflection_analysis": response,
            "timestamp": data.get("timestamp"),
            "patterns_identified": self._extract_patterns(historical_data)
        }
        
        self.save_to_memory(analysis_result)
        return analysis_result
    
    def _prepare_historical_data(self, data: Dict[str, Any]) -> str:
        """Prepare historical decisions and performance data for analysis"""
        historical_decisions = data.get("historical_decisions", [])
        
        if not historical_decisions:
            return "No historical data available for analysis."
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(historical_decisions)
        
        # Calculate basic statistics
        if not df.empty:
            success_rate = (df["outcome"] == "success").mean() if "outcome" in df.columns else 0
            avg_return = df["return"].mean() if "return" in df.columns else 0
            
            summary = f"""
            Total Decisions: {len(df)}
            Success Rate: {success_rate:.2%}
            Average Return: {avg_return:.2%}
            
            Recent Decisions:
            {df.tail(5).to_string()}
            """
            return summary
        
        return "Insufficient historical data for analysis."
    
    def _extract_patterns(self, historical_data: str) -> List[Dict[str, Any]]:
        """Extract recurring patterns from historical data"""
        # This would implement pattern recognition logic
        # For now, return a placeholder
        return [
            {
                "pattern_type": "market_condition",
                "frequency": "high",
                "success_rate": 0.75
            }
        ] 