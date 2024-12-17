from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
import re
import yfinance as yf
from agents import BaseAgent

class ReflectionAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.short_term_memory = []
        self.long_term_memory = []
        self.max_short_term_entries = 10
        self.last_transfer_time = datetime.now()
        self.transfer_interval = timedelta(days=30)

    def process_user_input(self, user_input: str) -> str:
        role = "You are an analyst assistant specialized in extracting stock tickers from user text."
        prompt = f"""Extract the stock ticker (e.g., AAPL, TSLA, MSFT) from the following user input:
        
        User Input: "{user_input}"

        If no clear ticker is provided, choose a default ticker like 'AAPL'.
        Just return the ticker symbol without explanation.
        """
        response = self._create_prompt(role, prompt)
        stock_symbol = re.search(r'[A-Z]{2,5}', response)
        if stock_symbol:
            return stock_symbol.group(0)
        else:
            return "AAPL"

    def fetch_stock_performance(self, stock_symbol: str) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = yf.download(stock_symbol, start=start_date, end=end_date, interval="1mo")

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df["monthly_return"] = df["Adj Close"].pct_change()
        df = df.dropna()[["Date", "Adj Close", "monthly_return"]]
        df.columns = ["timestamp", "price", "return"]
        return df

    def update_memory(self, stock_data: pd.DataFrame):
        if stock_data.empty:
            return

        for _, row in stock_data.iterrows():
            entry = {
                "timestamp": row["timestamp"].isoformat(),
                "return": row["return"],
                "price": row["price"]
            }
            self.short_term_memory.append(entry)

        self.check_and_transfer_memory()

    def summarize_memories(self) -> None:
        short_term_role = "You are a highly skilled financial memory summarizer agent specializing in recent (short-term) data analysis."
        short_term_prompt = f"""You are provided with the current short-term memory data related to recent financial observations and actions:
        
        Short-Term Memory Data:
        {self.short_term_memory}
        
        Instructions:
        1. Carefully review the listed short-term memory entries, which may include recent stock performance metrics, short-term trends, and any immediate insights.
        2. Summarize the key insights, patterns, or notable shifts observed in these entries.
        3. Your summary should be professional, concise, and focused, limited to 2-3 sentences. Avoid repetition and unnecessary details.
        
        Now, provide your concise and insightful summary:
        """
        
        short_summary = self._create_prompt(short_term_role, short_term_prompt).strip()
        self.short_term_memory.append({"summary": short_summary, "timestamp": datetime.now().isoformat()})
        
        long_term_role = "You are a highly skilled financial memory summarizer agent specializing in accumulated (long-term) data analysis."
        long_term_prompt = f"""You are provided with the current long-term memory data, which includes aggregated past summaries and historical insights:
        
        Long-Term Memory Data:
        {self.long_term_memory}
        
        Instructions:
        1. Review the long-term memory entries, which may contain overarching performance trends, previously identified patterns, and insights built up over time.
        2. Summarize the most important and enduring insights, patterns, or lessons learned from the accumulated data. Consider how these insights could inform future decisions or strategies.
        3. If there is no long-term data, simply return "No long-term data yet." Otherwise, provide 2-3 professional, concise sentences that capture the essence of the long-term memory.
        
        Now, provide your concise and insightful summary:
        """

        long_summary = self._create_prompt(long_term_role, long_term_prompt).strip()
        if long_summary.lower() != "no long-term data yet.":
            self.long_term_memory.append({"summary": long_summary, "timestamp": datetime.now().isoformat()})

    def analyze(self, user_input: str) -> Dict[str, Any]:
        stock_symbol = self.process_user_input(user_input)
        stock_data = self.fetch_stock_performance(stock_symbol)
        self.update_memory(stock_data)
        self.summarize_memories()
        recommendation = self._generate_recommendation_using_agent(stock_symbol)

        analysis_result = {
            "stock_symbol": stock_symbol,
            "recommendation": recommendation,
            "long_term_insights": self._summarize_long_term_memory(),
            "short_term_insights": self._summarize_short_term_memory()
        }
        return analysis_result

    def _generate_recommendation_using_agent(self, stock_symbol: str) -> str:
        role = "You are a seasoned financial advisor with extensive experience in equity markets, portfolio management, and risk assessment."
        prompt = f"""
        You have been provided with historical short-term and long-term memory data, reflecting previously observed market trends, performance metrics, strategic insights, and risk considerations. Your goal is to integrate and interpret these memories to provide a coherent, forward-looking recommendation for the specified stock.
    
        Contextual Data:
        - Short-Term Memory:
          {self.short_term_memory}
    
        - Long-Term Memory:
          {self.long_term_memory}
    
        The user is interested in the stock: {stock_symbol}.
    
        Instructions:
        1. Carefully review the provided short-term and long-term memory data.
        2. Identify any recurring patterns, notable shifts in performance, or key risk/reward signals that stand out from the integrated memory.
        3. Formulate a recommendation for the user on whether they should consider buying, holding, or selling shares of {stock_symbol}.
        4. Provide a concise, professional rationale supporting your recommendation, focusing on the most critical and enduring insights derived from the memory data.
        5. Keep your explanation clear, focused, and free of unnecessary jargon, while maintaining a professional and authoritative tone.
    
        Please provide your recommendation now:
        """
        response = self._create_prompt(role, prompt)
        return response.strip()

    def check_and_transfer_memory(self):
        now = datetime.now()
        if len(self.short_term_memory) >= self.max_short_term_entries or (now - self.last_transfer_time) > self.transfer_interval:
            self.transfer_to_long_term_memory()
            self.last_transfer_time = now

    def transfer_to_long_term_memory(self):
        if self.short_term_memory:
            role = "You are a highly experienced memory aggregation agent specializing in financial and strategic data integration."
            prompt = f"""
            You are given a series of short-term memory entries reflecting recent observations, actions, and insights. Your task is to review these entries and produce a concise yet comprehensive summary that captures the most significant trends, patterns, and insights. This summary will serve as a long-term reference, informing future analysis and decision-making.
    
            Context:
            Short-Term Memory Entries:
            {self.short_term_memory}
    
            Instructions:
            1. Carefully examine the provided short-term memory data for recurring themes, notable shifts, performance indicators, and any emergent patterns.
            2. Synthesize these observations into a cohesive long-term memory summary, focusing on the most critical and enduring insights rather than transient details.
            3. Keep the summary professional, clear, and actionable, limited to a short paragraph (2-4 sentences). The summary should be easily understandable for future reference.
    
            Please provide the aggregated long-term memory summary now:
            """
            aggregation = self._call_llm(role, prompt).strip()
    
            aggregated_entry = {
                "timestamp": datetime.now().isoformat(),
                "aggregated_results": self.short_term_memory,
                "aggregated_summary": aggregation
            }
            self.long_term_memory.append(aggregated_entry)
            self.short_term_memory = []

    def _summarize_long_term_memory(self) -> str:
        if not self.long_term_memory:
            return "No long-term insights available."
        return f"Stored {len(self.long_term_memory)} long-term memory entries."

    def _summarize_short_term_memory(self) -> str:
        if not self.short_term_memory:
            return "No short-term insights available."
        return f"Short-term memory has {len(self.short_term_memory)} entries."

    def save_to_memory(self, analysis_result: Dict[str, Any]):
        pass
