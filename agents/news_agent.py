from typing import Dict, Any, List
from newspaper import Article
import yfinance as yf
from agents import BaseAgent
from config import NEWS_SOURCES
from datetime import datetime
import pytz

class NewsAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sources = NEWS_SOURCES
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze news articles and market sentiment for given symbols
        
        Args:
            data: Dictionary containing symbols to analyze and timestamp
            
        Returns:
            Dictionary containing news analysis and sentiment
        """
        symbols = data.get("symbols", [])
        timestamp = data.get("timestamp")
        news_data = self._gather_news(symbols, timestamp)
        
        # Create prompt for news analysis
        role = """You are a sophisticated financial news analyst. Analyze the provided news 
        articles and extract key insights, sentiment, and potential market impact. Focus on 
        actionable trading implications. Consider the chronological order of events."""
        
        content = f"""Analyze the following news articles related to {', '.join(symbols)}:
        
        News Data:
        {news_data}
        
        Provide analysis in the following format:
        1. Overall Market Sentiment
        2. Key Events/Developments (in chronological order)
        3. Potential Market Impact
        4. Trading Implications
        
        Note: All news articles are from before {timestamp}.
        """
        
        response = self._create_prompt(role, content)
        
        analysis_result = {
            "news_analysis": response,
            "analyzed_symbols": symbols,
            "timestamp": timestamp,
            "has_news": bool(news_data.strip())
        }
        
        self.save_to_memory(analysis_result)
        return analysis_result
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to timezone-aware datetime object"""
        try:
            # Try parsing ISO format with timezone
            dt = datetime.fromisoformat(timestamp_str)
            if dt.tzinfo is None:
                # If no timezone info, assume UTC
                dt = dt.replace(tzinfo=pytz.UTC)
            return dt
        except ValueError:
            # If parsing fails, return current time in UTC
            return datetime.now(pytz.UTC)
    
    def _gather_news(self, symbols: List[str], timestamp: str = None) -> str:
        """
        Gather relevant news articles for the given symbols
        
        Args:
            symbols: List of stock symbols to gather news for
            timestamp: ISO format timestamp string. Only collect news before this time.
        """
        news_data = []
        cutoff_time = self._parse_timestamp(timestamp) if timestamp else datetime.now(pytz.UTC)
        
        for symbol in symbols:
            # Get stock info
            stock = yf.Ticker(symbol)
            # Get news
            try:
                news = stock.news  # Get all available news
                for article in news:
                    try:
                        # Convert Unix timestamp to UTC datetime
                        publish_time = datetime.fromtimestamp(
                            article.get("providerPublishTime", 0), 
                            tz=pytz.UTC
                        )
                        
                        # Skip articles published after the cutoff time
                        if publish_time >= cutoff_time:
                            continue
                            
                        # Parse article
                        news_article = Article(article.get("link"))
                        news_article.download()
                        news_article.parse()
                        
                        news_data.append({
                            "symbol": symbol,
                            "title": news_article.title,
                            "text": news_article.text[:500],  # First 500 chars
                            "source": article.get("source"),
                            "published": publish_time.isoformat()
                        })
                        
                        # Only keep the 5 most recent articles before cutoff
                        if len([n for n in news_data if n['symbol'] == symbol]) >= 5:
                            break
                            
                    except Exception as e:
                        print(f"Error parsing article for {symbol}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error fetching news for {symbol}: {str(e)}")
                continue
        
        # Sort news by publish time
        news_data.sort(key=lambda x: x['published'], reverse=True)
        
        # Format news data as string
        formatted_news = "\n\n".join([
            f"Symbol: {item['symbol']}\n"
            f"Title: {item['title']}\n"
            f"Source: {item['source']}\n"
            f"Published: {item['published']}\n"
            f"Summary: {item['text']}\n"
            for item in news_data
        ])
        
        return formatted_news