from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
from . import BaseAgent


class ReflectionAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.historical_decisions = []

        self.short_term_memory = []
        self.long_term_memory = []

        self.max_short_term_entries = 10
        self.last_transfer_time = datetime.now()
        self.transfer_interval = timedelta(days=1)

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        historical_data = self._prepare_historical_data(data)
        current_market = data.get("current_market", {})
        long_term_insights = self._summarize_long_term_memory()
        short_term_insights = self._summarize_short_term_memory()

        role = """You are a reflective trading analyst who learns from past decisions and 
        market patterns. Analyze historical trading decisions and current market conditions 
        to provide insights and recommendations."""

        content = f"""Analyze the following historical trading decisions and current market conditions:

        Long-Term Insights:
        {long_term_insights}

        Short-Term Insights:
        {short_term_insights}

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
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "patterns_identified": self._extract_patterns(historical_data),
            "current_market": current_market,
            "historical_summary": historical_data
        }

        self.save_to_short_term_memory(analysis_result)
        self.check_and_transfer_memory()

        return analysis_result

    def _prepare_historical_data(self, data: Dict[str, Any]) -> str:
        historical_decisions = data.get("historical_decisions", [])

        if not historical_decisions:
            return "No historical data available for analysis."

        df = pd.DataFrame(historical_decisions)

        if not df.empty:
            success_rate = (df["outcome"] == "success").mean(
            ) if "outcome" in df.columns else 0
            avg_return = df["return"].mean() if "return" in df.columns else 0

            summary = f"""
            Total Decisions: {len(df)}
            Success Rate: {success_rate:.2%}
            Average Return: {avg_return:.2%}

            Recent Decisions:
            {df.tail(5).to_string(index=False)}
            """
            return summary

        return "Insufficient historical data for analysis."

    def _extract_patterns(self, historical_data: str) -> List[Dict[str, Any]]:
        return [
            {
                "pattern_type": "market_condition",
                "frequency": "high",
                "success_rate": 0.75
            }
        ]

    def _create_prompt(self, role: str, content: str) -> str:
        prompt = f"{role}\n\n{content}"
        response = self._call_llm(prompt)
        return response

    def save_to_short_term_memory(self, analysis_result: Dict[str, Any]):
        self.short_term_memory.append(analysis_result)

    def check_and_transfer_memory(self):
        now = datetime.now()
        if len(self.short_term_memory) >= self.max_short_term_entries or (now - self.last_transfer_time) > self.transfer_interval:
            self.transfer_to_long_term_memory()
            self.last_transfer_time = now

    def transfer_to_long_term_memory(self):
        if self.short_term_memory:
            aggregated_entry = {
                "timestamp": datetime.now().isoformat(),
                "aggregated_results": self.short_term_memory
            }
            self.long_term_memory.append(aggregated_entry)
            self.short_term_memory = []
