from typing import Dict, Any, List
from agents import BaseAgent
from agents.debate_memory import MemorySummaryAgent
import re

class DebateAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.debate_rounds = 3
        self.roles = [
            {
                "name": "always_bull",
                "description": """You are an Extreme Bull Analyst who always chooses Bullish stance for each stock, regardless of conditions.
                                You must explicitly address (refute or justify) the points in the short-term memory from the previous round that contradict bullish stances or reinforce them."""
            },
            {
                "name": "always_bear",
                "description": """You are an Extreme Bear Analyst who always chooses Bearish stance for each stock, regardless of conditions.
                                You must explicitly address (refute or justify) the points in the short-term memory from the previous round that contradict bearish stances or reinforce them."""
            },
            {
                "name": "fundamental",
                "description": """You are a Fundamental Analyst focusing on macroeconomic indicators,
                                company financials, sector trends, and other fundamental factors.
                                In each round, you may choose Bullish or Bearish stance for each stock.
                                You must explicitly address (refute or justify) the points in the short-term memory from the previous round that contradict your stance or reinforce it."""
            },
            {
                "name": "technical",
                "description": """You are a Technical Analyst focusing on price trends, chart patterns,
                                technical indicators, and volume analysis.
                                In each round, you may choose Bullish or Bearish stance for each stock.
                                You must explicitly address (refute or justify) the points in the short-term memory from the previous round that contradict your stance or reinforce it."""
            },
            {
                "name": "risk",
                "description": """You are a Risk Analyst focusing on potential risks, uncertainties,
                                volatility, regulatory changes, and market sentiment shifts.
                                In each round, you may choose Bullish or Bearish stance for each stock.
                                You must explicitly address (refute or justify) the points in the short-term memory from the previous round that contradict your stance or reinforce it."""
            }
        ]
        self.memory_summarizer = MemorySummaryAgent(config=config)
        self.mid_term_memory = []
        self.short_term_memory = []
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
        debate_rounds = []
        stocks = sorted(market_data.keys())

        for round_num in range(self.debate_rounds):
            round_results = []
            for role_info in self.roles:
                if round_num == 0:
                    role = role_info["description"]
                else:
                    # 修改后的描述，每一轮都允许改变观点，并要求对上轮论点进行回应
                    role = f"""You are the {role_info['name']} analyst. Continue focusing on your domain.
                               You may change Bullish/Bearish stance freely this round.
                               You must explicitly refute or support previous round's differing opinions from the short-term memory.
                               If you maintain your previous stance, justify it against the arguments in short-term memory.
                               If you change your stance, explain why you changed in response to previous arguments.
                               """

                perspective_name = role_info["name"]

                stock_instructions = "\n".join(
                    [f"{i+1}. {symbol}: {market_data[symbol]}" for i, symbol in enumerate(stocks)]
                )

                # 修改的Prompt：在Instructions中明确要求参考short-term memory，并对上轮中与自己观点不符的观点进行反驳或解释。
                content = f"""
                Round {round_num + 1} of debate ({perspective_name.upper()}):

                Market Data for each stock:
                {stock_instructions}

                Proposed Action:
                {proposed_action}

                Mid-term Memory (accumulated):
                {self._get_mid_term_info()}

                Short-term Memory (last round only):
                {self._get_short_term_info()}

                Previous Arguments:
                {self._format_previous_rounds(debate_rounds)}

                Instructions:
                - For each stock listed above, choose either Bullish or Bearish stance. You may change your stance from previous rounds.
                - You must explicitly address (refute or justify) viewpoints from the Short-term Memory that contradict your stance or reinforce it.
                - Provide a short but specific viewpoint (4-5 sentences max) referencing these arguments.
                - Format the output so that for each stock you produce exactly one line:
                    "Stock: SYMBOL - Bullish(or Bearish) Your short viewpoint"
                - Do this in the same order as the stocks are listed.
                - Keep it concise and debate-like.
                """

                response = self._create_prompt(role, content)
                round_results.append({
                    "round": round_num + 1,
                    "perspective": perspective_name,
                    "arguments": response
                })

            debate_rounds.extend(round_results)

            # 从本轮的论点提取立场
            stock_stances = self._extract_stock_stances(round_results, stocks)

            # 移除原有的提前终止逻辑，不再检查是否所有观点一致
            # 原逻辑删除（下方是原位置的逻辑，现在已删除）

            last_round_num = round_results[-1]['round']
            this_round_data = [r['arguments'] for r in round_results if r['round'] == last_round_num]
            round_summary = self.memory_summarizer.summarize_speeches(this_round_data)
            self.short_term_memory.clear()
            self.memory_summarizer.add_to_short_term_memory(self.short_term_memory, round_summary)
            self.memory_summarizer.add_to_mid_term_memory(self.mid_term_memory, round_summary)

        return debate_rounds


    def _extract_stock_stances(self, round_data: List[Dict[str, Any]], stocks: List[str]) -> Dict[str, List[str]]:
        """Extract bullish/bearish stances for each stock from debate rounds"""
        stances_per_stock = {s: [] for s in stocks}
        
        for r in round_data:
            arguments = r['arguments'].lower().split('\n')
            for line in arguments:
                line = line.strip()
                
                # Look for stock signals in various formats
                for stock in stocks:
                    if stock.lower() in line:
                        # Check for bullish/bearish signals
                        if "bullish" in line:
                            stances_per_stock[stock].append("bullish")
                        elif "bearish" in line:
                            stances_per_stock[stock].append("bearish")
                        # If signal found, move to next line
                        break
        
        return stances_per_stock

    def _synthesize_debate(self, debate_rounds: List[Dict[str, Any]]) -> str:
        """Synthesize debate results into final analysis"""
        role = """You are a senior market strategist tasked with synthesizing insights 
                  from three specialized analysts (fundamental, technical, and risk) for multiple stocks."""
        
        # Extract final round stances
        last_round = [r for r in debate_rounds if r['round'] == max(r['round'] for r in debate_rounds)]
        stocks = set()
        for r in last_round:
            for line in r['arguments'].split('\n'):
                for stock in re.findall(r'[A-Z]{1,5}(?=[\s:,.]|$)', line):
                    stocks.add(stock)
        
        content = f"""
        Synthesize the following debate rounds into a final analysis:

        Mid-term Memory (accumulated from all rounds):
        {self._get_mid_term_info()}

        Short-term Memory (just last round):
        {self._get_short_term_info()}
        
        Debate History:
        {self._format_previous_rounds(debate_rounds)}
        
        Instructions:
        1. For each stock ({', '.join(sorted(stocks))}), clearly state if the consensus is BULLISH or BEARISH
        2. Format each stock signal as: "SYMBOL: SIGNAL" (e.g., "AAPL: BULLISH")
        3. Then provide your detailed analysis
        4. End with a clear "SIGNALS:" section listing all stock signals
        """
        
        return self._create_prompt(role, content)
    
    def _format_previous_rounds(self, debate_rounds: List[Dict[str, Any]]) -> str:
        if not debate_rounds:
            return "No previous arguments."
        
        formatted_rounds = []
        for round_data in debate_rounds:
            formatted_rounds.append(
                f"Round {round_data['round']} ({round_data['perspective'].upper()}):\n"
                f"{round_data['arguments']}\n"
            )
        
        return "\n".join(formatted_rounds)

    def _get_mid_term_info(self) -> str:
        if not self.mid_term_memory:
            return "No mid-term memory recorded."
        return " | ".join(self.mid_term_memory)

    def _get_short_term_info(self) -> str:
        if not self.short_term_memory:
            return "No short-term memory recorded."
        return self.short_term_memory[-1]

    def _calculate_confidence(self, debate_rounds: List[Dict[str, Any]], market_data: Dict[str, Any]) -> float:
        confidence = 0.5
        trend = market_data.get("trend", "flat")

        if trend == "up":
            confidence += 0.1
        elif trend == "down":
            confidence -= 0.05
        else:
            confidence -= 0.02

        fundamental_args = [r['arguments'] for r in debate_rounds if r['perspective'] == 'fundamental']
        technical_args   = [r['arguments'] for r in debate_rounds if r['perspective'] == 'technical']
        risk_args        = [r['arguments'] for r in debate_rounds if r['perspective'] == 'risk']

        all_fundamental_text = " ".join(fundamental_args).lower()
        all_technical_text = " ".join(technical_args).lower()
        all_risk_text = " ".join(risk_args).lower()

        agreements = 0
        disagreements = 0

        def check_pairwise(text_a: str, text_b: str):
            local_agreements = 0
            local_disagreements = 0

            if "mild increase" in text_a and "mild increase" in text_b:
                local_agreements += 1
            if "low risk" in text_a and "low risk" in text_b:
                local_agreements += 1
            elif ("low risk" in text_a and "high risk" in text_b) or ("high risk" in text_a and "low risk" in text_b):
                local_disagreements += 1
            if "downward pressure" in text_a and "upward momentum" in text_b:
                local_disagreements += 1
            elif "downward pressure" in text_b and "upward momentum" in text_a:
                local_disagreements += 1

            return local_agreements, local_disagreements

        a, d = check_pairwise(all_fundamental_text, all_technical_text)
        agreements += a
        disagreements += d

        a, d = check_pairwise(all_fundamental_text, all_risk_text)
        agreements += a
        disagreements += d

        a, d = check_pairwise(all_technical_text, all_risk_text)
        agreements += a
        disagreements += d

        confidence += (agreements * 0.05)
        confidence -= (disagreements * 0.05)

        if ("moving average crossing above price" in all_fundamental_text and
            "moving average crossing above price" in all_technical_text) or \
           ("moving average crossing above price" in all_fundamental_text and
            "moving average crossing above price" in all_risk_text) or \
           ("moving average crossing above price" in all_technical_text and
            "moving average crossing above price" in all_risk_text):
            confidence += 0.03

        confidence = max(0.0, min(1.0, confidence))
        return confidence
