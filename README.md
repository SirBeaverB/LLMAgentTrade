# Multi-Agent Quants 🤖📈

![Multi-Agent Quants Interface](utils/icon.ico)

A sophisticated multi-agent trading system powered by LangChain that uses different specialized AI agents to make informed trading decisions. The system features a modern, dark-themed web interface built with Streamlit.

## Live Demo

Access the deployed version at: [Multi-Agent Quants Web App](https://multi-agent-quants.streamlit.app/)

## Features

### Multi-Agent System
1. **News-Driven Agent**: Analyzes current news and market sentiment
2. **Reflection-Driven Agent**: Analyzes historical decisions and patterns
3. **Debate-Driven Agent**: Creates pros and cons analysis through internal debate
4. **Decision Coordinator Agent**: Aggregates insights from other agents to make final decisions

### User Interface
- Modern web interface
- Real-time stock price charts
- Interactive configuration settings
- Live agent analysis updates
- Customizable trading parameters

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/ZhengxuYan/Multi-Agent-Quants.git
cd Multi-Agent-Quants
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python -m streamlit run app.py
```

5. Open your browser and navigate to:
```
http://localhost:8501
```

## Usage

1. Enter your OpenAI API key in the sidebar
2. Configure model settings for each agent:
   - Model selection (GPT-3.5, GPT-4)
   - Temperature
   - Max tokens
3. Adjust trading settings:
   - Analysis timeframe
   - Risk tolerance
   - News sources
4. Select stocks to analyze (up to 5)
5. Click "Run Analysis" to start the analysis

## Project Structure

```
LLMAgentTrade/
├── agents/
│   ├── __init__.py
│   ├── news_agent.py
│   ├── reflection_agent.py
│   ├── debate_agent.py
│   └── coordinator_agent.py
├── config.py
├── main.py
├── app.py
└── requirements.txt
```

## Configuration

All settings can be adjusted through the web interface:
- Model parameters for each agent
- Trading timeframes and risk tolerance
- News sources for analysis
- Stock symbol selection

## Requirements

- Python 3.10
- OpenAI API key
- Required packages listed in requirements.txt

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.