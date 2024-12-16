from typing import Dict, Any, List
from newspaper import Article
import yfinance as yf
from agents import BaseAgent
from config import NEWS_SOURCES

class NewsAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sources = NEWS_SOURCES
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze news articles and market sentiment for given symbols
        
        Args:
            data: Dictionary containing symbols to analyze
            
        Returns:
            Dictionary containing news analysis and sentiment
        """
        symbols = data.get("symbols", [])
        news_data = self._gather_news(symbols)
        
        # Create prompt for news analysis
        role = """You are a sophisticated financial news analyst. Analyze the provided news 
        articles and extract key insights, sentiment, and potential market impact. Focus on 
        actionable trading implications."""
        
        content = f"""Analyze the following news articles related to {', '.join(symbols)}:
        
        News Data:
        {news_data}
        
        Provide analysis in the following format:
        1. Overall Market Sentiment
        2. Key Events/Developments
        3. Potential Market Impact
        4. Trading Implications
        """
        
        response = self._create_prompt(role, content)
        
        analysis_result = {
            "news_analysis": response,
            "analyzed_symbols": symbols,
            "timestamp": data.get("timestamp")
        }
        
        self.save_to_memory(analysis_result)
        return analysis_result
    
    def _gather_news(self, symbols: List[str]) -> str:
        """Gather relevant news articles for the given symbols"""
        news_data = []
        
        for symbol in symbols:
            # Get stock info
            stock = yf.Ticker(symbol)
            # Get news
            try:
                news = stock.news[:5]  # Get latest 5 news items
                for article in news:
                    try:
                        # Parse article
                        news_article = Article(article.get("link"))
                        news_article.download()
                        news_article.parse()
                        
                        news_data.append({
                            "symbol": symbol,
                            "title": news_article.title,
                            "text": news_article.text[:500],  # First 500 chars
                            "source": article.get("source"),
                            "published": article.get("providerPublishTime")
                        })
                    except Exception as e:
                        continue
            except Exception as e:
                continue
        
        # Format news data as string
        formatted_news = "\n\n".join([
            f"Symbol: {item['symbol']}\n"
            f"Title: {item['title']}\n"
            f"Source: {item['source']}\n"
            f"Summary: {item['text']}\n"
            for item in news_data
        ])
        
        return formatted_news 