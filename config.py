import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Free tier configuration using Ollama
FREE_TIER_SETTINGS = {
    "model": "llama2",  # Default free model
    "temperature": 0.7,
    "max_tokens": 1000
}

# Agent Configuration
AGENT_SETTINGS = {
    "news_agent": {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1500
    },
    "reflection_agent": {
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "max_tokens": 1500
    },
    "debate_agent": {
        "model": "gpt-4o-mini",
        "temperature": 0.8,
        "max_tokens": 2000
    },
    "coordinator_agent": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 1000
    }
}

# Available models
AVAILABLE_MODELS = {
    "premium": ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini", "gpt-4o"],
    "free": ["llama2", "mistral", "codellama", "phi"]
}

# News Sources Configuration
NEWS_SOURCES = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com"
]

# Trading Configuration
TRADING_SETTINGS = {
    "analysis_timeframe": "1d",  # 1 day analysis window
    "risk_tolerance": 0.02       # Maximum risk per trade (2% of portfolio)
} 