import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Free tier configuration using HuggingFace
FREE_TIER_SETTINGS = {
    "model": "gpt2",  # Default to most reliable free model
    "temperature": 0.7,
    "max_tokens": 512,  # Reduced max tokens to stay within model limits
    "warning": """
        NOTE: You are using the free tier with HuggingFace models.
        These models have significantly reduced performance compared to premium models
        and should mainly be used for testing purposes.
        For production use, please use the premium tier with OpenAI models.
        
        Note: Some models like Mistral-7B-Instruct require a HuggingFace Pro 
        subscription and explicit model access approval.
    """
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
        "temperature": 0.7,
        "max_tokens": 1500
    },
    "debate_agent": {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "coordinator_agent": {
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "max_tokens": 1000
    }
}

# Available models
AVAILABLE_MODELS = {
    "premium": ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini", "gpt-4o"],
    "free": [
        # WARNING: Free tier models have significantly reduced performance
        # and are mainly suitable for testing purposes. For production use,
        # please use the premium tier with OpenAI models.
        "gpt2",                                # Most reliable but basic
        "distilgpt2",                         # Faster but less capable
        "bigscience/bloom-560m",              # Medium model, mixed results
        "EleutherAI/pythia-160m"              # Small but consistent
    ]
}

# News Sources Configuration
NEWS_SOURCES = [
    "yfinance",
    "alpha_vantage",
    "finnhub",
    "newsapi",
    "sec"
]

# Trading Configuration
TRADING_SETTINGS = {
    "analysis_timeframe": "1d",  # 1 day analysis window
    "risk_tolerance": 0.02       # Maximum risk per trade (2% of portfolio)
} 