from typing import Dict, Any, List
from agents import BaseAgent
from agents.debate_memory import MemorySummaryAgent

class DebateAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.debate_rounds = 3
        self.roles = [
            {
                "name": "fundamental",
                "description": """You are a Fundamental Analyst focusing on macroeconomic indicators,
                                company financials, sector trends, and other fundamental factors.
                                You can choose your stance (Bullish or Bearish) freely for each stock each round.
                                Argue briefly and directly."""
            },
            {
                "name": "technical",
                "description": """You are a Technical Analyst focusing on price trends, chart patterns,
                                technical indicators, and volume analysis.
                                You can choose your stance (Bullish or Bearish) freely for each stock each round.
                                Argue briefly and directly."""
            },
            {
                "name": "risk",
                "description": """You are a Risk Analyst focusing on potential risks, uncertainties,
                                volatility, regulatory changes, and market sentiment shifts.
                                You can choose your stance (Bullish or Bearish) freely for each stock each round.
                                Argue briefly and directly."""
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
                    role = f"You are the {role_info['name']} analyst. Continue focusing on your domain. Remember you can choose Bullish or Bearish freely for each stock."
                perspective_name = role_info["name"]

                stock_instructions = "\n".join(
                    [f"{i+1}. {symbol}: {market_data[symbol]}" for i, symbol in enumerate(stocks)]
                )

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
                - For each stock listed above, choose either Bullish or Bearish stance.
                - Then provide a short viewpoint (1-2 sentences max) for that stock.
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

            stock_stances = self._extract_stock_stances(round_results, stocks)

            all_same_for_all_stocks = True
            for s in stocks:
                s_stance_set = set(stock_stances[s])
                if len(s_stance_set) > 1:
                    all_same_for_all_stocks = False
               

                if all_same_for_all_stocks:
                    break

        
            last_round_num = round_results[-1]['round']
            this_round_data = [r['arguments'] for r in round_results if r['round'] == last_round_num]
            round_summary = self.memory_summarizer.summarize_speeches(this_round_data)
            self.short_term_memory.clear()
            self.memory_summarizer.add_to_short_term_memory(self.short_term_memory, round_summary)
            self.memory_summarizer.add_to_mid_term_memory(self.mid_term_memory, round_summary)

        return debate_rounds


    def _extract_stock_stances(self, round_data: List[Dict[str, Any]], stocks: List[str]) -> Dict[str, List[str]]:
        stances_per_stock = {s: [] for s in stocks}

        for r in round_data:
            arguments = r['arguments'].strip().split('\n')
            for line in arguments:
                line = line.strip().lower()
                if line.startswith("stock:"):
                    parts = line.split('-')
                    if len(parts) < 2:
                        continue
                    symbol_part = parts[0].replace("stock:", "").strip()
                    stance_part = parts[1].strip()
                    symbol = symbol_part
                    if symbol in stances_per_stock:
                        if stance_part.startswith("bullish"):
                            stances_per_stock[symbol].append("bullish")
                        elif stance_part.startswith("bearish"):
                            stances_per_stock[symbol].append("bearish")
                        else:
                            stances_per_stock[symbol].append("neutral")

        return stances_per_stock

    def _synthesize_debate(self, debate_rounds: List[Dict[str, Any]]) -> str:
        role = """You are a senior market strategist tasked with synthesizing insights 
                    from three specialized analysts (fundamental, technical, and risk) for multiple stocks."""
        
        content = f"""
        Synthesize the following debate rounds into a final analysis:

        Mid-term Memory (accumulated from all rounds):
        {self._get_mid_term_info()}

        Short-term Memory (just last round):
        {self._get_short_term_info()}
        
        Debate History:
        {self._format_previous_rounds(debate_rounds)}
        
        Instructions:
        - Provide a balanced final analysis for each stock considered.
        - For each stock, mention if there was more Bullish or Bearish consensus.
        - Provide a final recommendation considering all perspectives and risk/reward.
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
