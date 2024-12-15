# LLMAgentTrade

A multi-agent trading system powered by LangChain that uses different specialized agents to make informed trading decisions.

## Agents

1. **News-Driven Agent**: Analyzes current news and market sentiment
2. **Reflection-Driven Agent**: Analyzes historical decisions and patterns
3. **Debate-Driven Agent**: Creates pros and cons analysis through internal debate
4. **Decision Coordinator Agent**: Aggregates insights from other agents to make final decisions

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
LLMAgentTrade/
├── agents/
│   ├── __init__.py
│   ├── news_agent.py
│   ├── reflection_agent.py
│   ├── debate_agent.py
│   └── coordinator_agent.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── config.py
├── main.py
└── requirements.txt
```

## Usage

Run the trading system:
```bash
python main.py
```