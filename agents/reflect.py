from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
from . import BaseAgent
import re


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
        patterns = []
        success_rate_match = re.search(
            r"Success Rate:\s*([\d\.]+)%", historical_data)
        avg_return_match = re.search(
            r"Average Return:\s*([\d\.]+)%", historical_data)

        success_rate = float(success_rate_match.group(1)
                             ) if success_rate_match else 0.0
        avg_return = float(avg_return_match.group(1)
                           ) if avg_return_match else 0.0
        if success_rate > 70:
            patterns.append({
                "pattern_type": "strong_performance",
                "metric": "success_rate",
                "value": success_rate,
                "description": "Historical success rate is above 70%, indicating strong performance."
            })

        if avg_return > 5:
            patterns.append({
                "pattern_type": "high_average_return",
                "metric": "avg_return",
                "value": avg_return,
                "description": "Average return is above 5%, indicating consistently good outcomes."
            })

        recent_lines = []
        start_parsing = False
        for line in historical_data.split('\n'):
            line = line.strip()
            if "Recent Decisions:" in line:
                start_parsing = True
                continue
            if start_parsing and line:
                recent_lines.append(line)

        success_count = 0
        for dec_line in recent_lines:
            parts = dec_line.split()
            if len(parts) >= 3:
                outcome = parts[2].lower()
                if outcome == "success":
                    success_count += 1
                else:
                    success_count = 0
            if success_count >= 3:
                patterns.append({
                    "pattern_type": "streak_of_successes",
                    "streak_length": success_count,
                    "description": f"Found a streak of {success_count} consecutive successful decisions."
                })
                success_count = 0

        if not patterns:
            patterns.append({
                "pattern_type": "no_significant_pattern",
                "description": "No significant patterns detected based on the given historical data."
            })

        return patterns

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

    def _summarize_long_term_memory(self) -> str:
        if not self.long_term_memory:
            return "No long-term insights available."
        summary = "Long-Term Memory Entries:\n"
        for entry in self.long_term_memory[-3:]:
            summary += f"- {entry['timestamp']
                            } with {len(entry['aggregated_results'])} items\n"
        return summary

    def _summarize_short_term_memory(self) -> str:
        if not self.short_term_memory:
            return "No short-term insights available."
        summary = f"Short-Term Memory has {
            len(self.short_term_memory)} entries.\n"
        last_entry = self.short_term_memory[-1]
        summary += f"Last entry reflection timestamp: {
            last_entry['timestamp']}\n"
        summary += f"Last entry reflection content excerpt: {
            last_entry['reflection_analysis'][:100]}...\n"
        return summary
