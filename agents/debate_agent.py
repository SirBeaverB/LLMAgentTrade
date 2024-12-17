from typing import Dict, Any, List
from agents import BaseAgent
from agents.debate_memory import MemorySummaryAgent

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
        news_analysis = data.get("news_analysis", {})
        reflection_analysis = data.get("reflection_analysis", {})

        # Add context to roles based on other agents' analyses
        enhanced_roles = []
        for role in self.roles:
            role_info = role.copy()
            
            # Add context to role description
            context_additions = f"""
            Consider the following additional context in your analysis:
            
            News Analysis:
            {news_analysis}
            
            Reflection Analysis:
            {reflection_analysis}
            """
            
            role_info["description"] = role_info["description"] + context_additions
            enhanced_roles.append(role_info)
        
        # Use enhanced roles for debate
        debate_results = self._conduct_debate(market_data, proposed_action, enhanced_roles)
        
        final_analysis = self._synthesize_debate(debate_results)
        
        analysis_result = {
            "debate_analysis": final_analysis,
            "debate_rounds": debate_results,
            "timestamp": data.get("timestamp"),
            "confidence_score": self._calculate_confidence(debate_results, market_data)
        }
        
        self.save_to_memory(analysis_result)
        return analysis_result
    
    def _conduct_debate(self, market_data: Dict[str, Any], proposed_action: Dict[str, Any], 
                       enhanced_roles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        debate_rounds = []
        stocks = sorted(market_data.keys())

        for round_num in range(self.debate_rounds):
            round_results = []
            for role_info in enhanced_roles:
                if round_num == 0:
                    role = role_info["description"]
                else:
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
                - Keep it concise and debate-like. Use adjective-rich language to convey confidence and expertise.
                """

                response = self._create_prompt(role, content)
                round_results.append({
                    "round": round_num + 1,
                    "perspective": perspective_name,
                    "arguments": response
                })

            debate_rounds.extend(round_results)

            stock_stances = self._extract_stock_stances(round_results, stocks)

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
                    from five specialized analysts (fundamental, technical, risk, always_bull, always_bear) for multiple stocks."""
        
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
        - Provide a final recommendation considering all perspectives, including the extreme bull/bear, and overall risk/reward.
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

        if not debate_rounds:
            return confidence

        total_rounds = max(r["round"] for r in debate_rounds)
        expert_roles = ["fundamental", "technical", "risk"]
        
        sum_of_weights = total_rounds * (total_rounds + 1) / 2.0
        increments = 0.0 

        domain_positive_terms = [
            'strong momentum', 'investor confidence', 'reinforcing a positive outlook', 'upward momentum', 
            'robust interest', 'healthy demand', 'resilience', 'favorable outlook', 'clearly', 'strongly', 
            'definitely', 'consistently', 'well-supported', 'firm evidence', 'high conviction', 'no doubt', 
            'strongly grounded', 'robust evidence', 'unwavering', 'highly credible', 'solid foundation',
            'conclusive', 'authoritative', 'well-substantiated'
        ]

        domain_negative_terms = [
            'regulatory scrutiny', 'downward movement', 'selling pressure', 'challenges', 'volatility', 'pressure',
            'possibly', 'might', 'unclear', 'uncertain', 'tentative', 'questionable', 'ambiguous', 'unverified', 
            'guesswork', 'speculation', 'lack of clarity', 'insufficient data', 'doubtfully', 'not guaranteed', 
            'inconclusive', 'skeptical', 'dubious', 'no clear evidence'
        ]


        for round_num in range(1, total_rounds + 1):
            current_round_experts = [r for r in debate_rounds if r["round"] == round_num and r["perspective"] in expert_roles]

            if len(current_round_experts) < 3:
                continue

            stock_stances = {}
            all_arguments_text = [] 
            
            for entry in current_round_experts:
                perspective = entry["perspective"]
                lines = entry["arguments"].strip().split('\n')
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped.lower().startswith("stock:"):
                        parts = line_stripped.split('-', 1)
                        if len(parts) != 2:
                            continue
                        stock_part = parts[0].replace("Stock:", "").strip()
                        stance_part = parts[1].strip().lower()
                        if stance_part.startswith("bullish"):
                            stance = "bullish"
                        elif stance_part.startswith("bearish"):
                            stance = "bearish"
                        else:
                            stance = "neutral"

                        if stock_part not in stock_stances:
                            stock_stances[stock_part] = {}
                        stock_stances[stock_part][perspective] = stance
                    all_arguments_text.append(line_stripped.lower())

            round_weight = round_num
            for symbol, stances in stock_stances.items():
                if len(stances) == 3:
                    f_stance = stances.get("fundamental", "neutral")
                    t_stance = stances.get("technical", "neutral")
                    r_stance = stances.get("risk", "neutral")
                    if f_stance == t_stance == r_stance and f_stance in ["bullish", "bearish"]:
                        increments += 0.1 * round_weight
                    else:
                        increments -= 0.05 * round_weight
            
            text_joined = " ".join(all_arguments_text)
            domain_positive_count = sum(term in text_joined for term in domain_positive_terms)
            domain_negative_count = sum(term in text_joined for term in domain_negative_terms)
            increments += 0.005 * domain_positive_count * round_weight
            increments -= 0.005 * domain_negative_count * round_weight

        confidence += increments / sum_of_weights

        confidence = max(0.0, min(1.0, confidence))
        return confidence

