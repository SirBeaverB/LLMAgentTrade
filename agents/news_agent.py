from typing import Dict, Any, List
from newspaper import Article
import yfinance as yf
from agents import BaseAgent
from config import NEWS_SOURCES
from datetime import datetime, timedelta
import pytz
import requests
import json
import os
from bs4 import BeautifulSoup
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class NewsAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # All available sources
        self.available_sources = {
            "yfinance": self._get_yfinance_news,
            "alpha_vantage": self._get_alpha_vantage_news,
            "finnhub": self._get_finnhub_data,
            "newsapi": self._get_newsapi_articles,
            "sec": self._get_sec_filings
        }
        
        # Default enabled sources (can be overridden)
        self.enabled_sources = config.get("enabled_sources", list(self.available_sources.keys()))
        
        # Initialize session for better performance
        self.session = requests.Session()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # API Keys
        self.alpha_vantage_key = "EJSMMZ5KEBMB24DM"
        self.finnhub_key = "ctgii41r01qn78n41m20ctgii41r01qn78n41m2g"
        self.newsapi_key = "a706f78ed86742b286ce94206f2dcdd1"
        
        # Set default lookback window (in days)
        self.lookback_days = config.get('lookback_days', 5)
        self.logger.info(f"Initialized NewsAgent with {self.lookback_days} days lookback window")
        
        # Validate Finnhub API key with a simple request
        try:
            test_url = "https://finnhub.io/api/v1/stock/symbol?exchange=US"
            test_response = self.session.get(
                test_url,
                headers={'X-Finnhub-Token': self.finnhub_key}
            )
            
            if test_response.status_code == 401:
                self.logger.error("Invalid Finnhub API key")
                self.finnhub_key = None
            elif test_response.status_code == 429:
                self.logger.warning("Finnhub API rate limit reached during validation")
            elif test_response.status_code != 200:
                self.logger.warning(f"Finnhub API test returned status {test_response.status_code}")
                self.logger.debug(f"Test URL: {test_url}")
                self.logger.debug(f"Response: {test_response.text}")
        except Exception as e:
            self.logger.error(f"Error validating Finnhub API key: {str(e)}")
            self.finnhub_key = None
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze news articles and market sentiment for given symbols
        
        Args:
            data: Dictionary containing:
                - symbols: List of symbols to analyze
                - timestamp: Current timestamp
                - lookback_days: Optional, override default lookback window
            
        Returns:
            Dictionary containing news analysis and sentiment
        """
        symbols = data.get("symbols", [])
        timestamp = data.get("timestamp")
        
        # Update enabled sources if provided
        if "enabled_sources" in data:
            self.enabled_sources = data["enabled_sources"]
            self.logger.info(f"Using news sources: {', '.join(self.enabled_sources)}")
        
        # Allow override of lookback window per analysis
        lookback_days = data.get("lookback_days", self.lookback_days)
        
        # Calculate cutoff time based on lookback window
        if isinstance(timestamp, (str, datetime)):
            if isinstance(timestamp, str):
                try:
                    current_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    self.logger.warning(f"Invalid timestamp format: {timestamp}, using current time")
                    current_time = datetime.now(pytz.UTC)
            else:
                current_time = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=pytz.UTC)
        else:
            current_time = datetime.now(pytz.UTC)
        
        cutoff_time = current_time - timedelta(days=lookback_days)
        self.logger.info(f"Analyzing news from {cutoff_time} to {current_time}")
        
        # Gather news from multiple sources concurrently
        news_data = self._gather_comprehensive_news(symbols, cutoff_time, current_time)
        
        # Create prompt for news analysis
        role = """You are a sophisticated financial news analyst. Analyze the provided news 
        articles and extract key insights, sentiment, and potential market impact. Focus on 
        actionable trading implications. Consider:
        1. Official news sources
        2. Market sentiment indicators
        3. Social media sentiment
        4. SEC filings and corporate announcements
        5. Technical indicators
        
        Weight the sources appropriately, giving more importance to official announcements 
        and verified news sources."""
        
        content = f"""Analyze the following comprehensive news data for {', '.join(symbols)}:
        
        News Data:
        {news_data['articles']}
        
        Market Sentiment Indicators:
        {news_data['sentiment']}
        
        SEC Filings and Corporate Announcements:
        {news_data['sec_filings']}
        
        Technical Indicators:
        {news_data['technical']}
        
        Provide analysis in the following format:
        1. Overall Market Sentiment (with confidence level)
        2. Key Events/Developments (in chronological order)
        3. Potential Market Impact (short-term and long-term)
        4. Trading Implications
        5. Risk Factors
        
        Note: All data is from before {timestamp}.
        """
        
        response = self._create_prompt(role, content)
        
        analysis_result = {
            "news_analysis": response,
            "analyzed_symbols": symbols,
            "timestamp": timestamp,
            "has_news": bool(news_data['articles'].strip()),
            "sentiment_scores": news_data['sentiment_scores'],
            "source_breakdown": news_data['source_breakdown'],
            "confidence_score": self._calculate_confidence_score(news_data)
        }
        
        self.save_to_memory(analysis_result)
        return analysis_result
    
    def _gather_comprehensive_news(self, symbols: List[str], timestamp: str, current_time: datetime) -> Dict[str, Any]:
        """Gather news from multiple sources concurrently"""
        try:
            # Ensure timestamp is a datetime object
            if isinstance(timestamp, str):
                cutoff_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, datetime):
                cutoff_time = timestamp
            else:
                self.logger.warning(f"Invalid timestamp type: {type(timestamp)}, using current time minus lookback")
                cutoff_time = current_time - timedelta(days=self.lookback_days)
            
            # Ensure timezone awareness
            if cutoff_time.tzinfo is None:
                cutoff_time = cutoff_time.replace(tzinfo=pytz.UTC)
            
        except Exception as e:
            self.logger.error(f"Error parsing timestamp: {str(e)}")
            cutoff_time = current_time - timedelta(days=self.lookback_days)
        
        # Initialize result structure
        news_data = {
            'articles': "",
            'sentiment': "",
            'sec_filings': "",
            'technical': "",
            'sentiment_scores': {},
            'source_breakdown': {}
        }
        
        # Create tasks for concurrent execution
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            for symbol in symbols:
                # Only gather news from enabled sources
                for source_name in self.enabled_sources:
                    if source_name in self.available_sources:
                        futures.append(
                            executor.submit(
                                self.available_sources[source_name], 
                                symbol, 
                                cutoff_time,
                                current_time
                            )
                        )
            
            # Collect results
            all_articles = []
            sentiment_data = []
            filings_data = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['type'] == 'news':
                        all_articles.extend(result['data'])
                    elif result['type'] == 'sentiment':
                        sentiment_data.append(result['data'])
                    elif result['type'] == 'filing':
                        filings_data.extend(result['data'])
                except Exception as e:
                    self.logger.error(f"Error collecting news data: {str(e)}")
        
        # Process and format the collected data
        news_data['articles'] = self._format_articles(all_articles)
        news_data['sentiment'] = self._format_sentiment(sentiment_data)
        news_data['sec_filings'] = self._format_filings(filings_data)
        news_data['technical'] = self._get_technical_indicators(symbols)
        news_data['sentiment_scores'] = self._calculate_sentiment_scores(all_articles, sentiment_data)
        news_data['source_breakdown'] = self._calculate_source_breakdown(all_articles)
        
        return news_data
    
    def _get_yfinance_news(self, symbol: str, cutoff_time: datetime, current_time: datetime) -> Dict[str, Any]:
        """Get news from YFinance"""
        try:
            stock = yf.Ticker(symbol)
            articles = []
            
            for article in stock.news:
                try:
                    # Ensure timezone-aware datetime
                    publish_time = datetime.fromtimestamp(
                        article.get("providerPublishTime", 0)
                    ).replace(tzinfo=pytz.UTC)
                    
                    # Changed condition and ensure both times are timezone-aware
                    if cutoff_time.replace(tzinfo=pytz.UTC) <= publish_time <= current_time.replace(tzinfo=pytz.UTC):
                        news_article = Article(article.get("link"))
                        news_article.download()
                        news_article.parse()
                        
                        articles.append({
                            "symbol": symbol,
                            "title": news_article.title,
                            "text": news_article.text[:500],
                            "source": "YFinance",
                            "published": publish_time.isoformat(),
                            "url": article.get("link"),
                            "sentiment": None
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error parsing YFinance article: {str(e)}")
                    continue
            return {"type": "news", "data": articles}
            
        except Exception as e:
            self.logger.error(f"Error fetching YFinance news: {str(e)}")
            return {"type": "news", "data": []}
    
    def _get_alpha_vantage_news(self, symbol: str, cutoff_time: datetime, current_time: datetime) -> Dict[str, Any]:
        """Get news from Alpha Vantage"""
        try:
            # Add limit parameter to reduce unnecessary data transfer
            limit = min(500, self.lookback_days * 100)  # Estimate 100 articles per day
            url = (f"https://www.alphavantage.co/query"
                   f"?function=NEWS_SENTIMENT"
                   f"&tickers={symbol}"
                   f"&limit={limit}"
                   f"&apikey={self.alpha_vantage_key}")
            
            response = self.session.get(url)
            data = response.json()
            
            articles = []
            
            # Ensure cutoff_time and current_time are timezone-aware
            cutoff_time = cutoff_time.replace(tzinfo=pytz.UTC) if cutoff_time.tzinfo is None else cutoff_time
            current_time = current_time.replace(tzinfo=pytz.UTC) if current_time.tzinfo is None else current_time
            
            for item in data.get('feed', []):
                try:
                    # Handle different timestamp formats
                    time_str = item.get('time_published', '')
                    try:
                        # Try parsing Alpha Vantage's format (YYYYMMDDTHHmmss)
                        publish_time = datetime.strptime(time_str, '%Y%m%dT%H%M%S')
                    except ValueError:
                        try:
                            # Fallback to ISO format
                            publish_time = datetime.fromisoformat(time_str.replace('T', ' '))
                        except ValueError:
                            self.logger.warning(f"Unable to parse timestamp: {time_str}")
                            continue
                    
                    # Make publish_time timezone-aware
                    publish_time = publish_time.replace(tzinfo=pytz.UTC)
                    
                    # Compare timezone-aware datetimes
                    if cutoff_time <= publish_time <= current_time:
                        article = {
                            "symbol": symbol,
                            "title": item.get('title', ''),
                            "text": item.get('summary', ''),
                            "source": "Alpha Vantage",
                            "published": publish_time.isoformat(),
                            "url": item.get('url', ''),
                            "sentiment": item.get('overall_sentiment_score')
                        }
                        articles.append(article)
                        
                except Exception as e:
                    self.logger.error(f"Error processing Alpha Vantage news item: {str(e)}")
                    continue
            
            self.logger.info(f"Alpha Vantage articles: {articles}")
            return {"type": "news", "data": articles}
            
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage news: {str(e)}")
            return {"type": "news", "data": []}
    
    def _get_finnhub_data(self, symbol: str, cutoff_time: datetime, current_time: datetime) -> Dict[str, Any]:
        """Get news and sentiment from Finnhub"""
        try:
            headers = {
                'X-Finnhub-Token': self.finnhub_key
            }
            
            # Use cutoff_time as from_date and current time as to_date
            to_date = current_time
            from_date = cutoff_time
            
            # Get company news with proper date range
            news_url = (f"https://finnhub.io/api/v1/company-news"
                       f"?symbol={symbol}"
                       f"&from={from_date.strftime('%Y-%m-%d')}"
                       f"&to={to_date.strftime('%Y-%m-%d')}")
            
            news_response = self.session.get(news_url, headers=headers)
            
            # Check response status
            if news_response.status_code == 422:
                self.logger.error(f"Finnhub API key invalid or request malformed for {symbol}")
                self.logger.debug(f"Request URL: {news_url}")
                self.logger.debug(f"Response: {news_response.text}")
                return {"type": "sentiment", "data": {"symbol": symbol}}
            elif news_response.status_code == 429:
                self.logger.error(f"Finnhub API rate limit exceeded for {symbol}")
                return {"type": "sentiment", "data": {"symbol": symbol}}
            elif news_response.status_code != 200:
                self.logger.error(f"Finnhub news API returned status {news_response.status_code} for {symbol}")
                return {"type": "sentiment", "data": {"symbol": symbol}}
            elif news_response.status_code == 403:
                self.logger.error(f"Invalid Finnhub API key or subscription level doesn't allow sentiment access")
                self.finnhub_key = None  # Invalidate the key
                return {"type": "sentiment", "data": {"symbol": symbol}}
            
            try:
                news_data = news_response.json()
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode Finnhub news response for {symbol}: {str(e)}")
                news_data = []
            
            # Get sentiment data
            sentiment_url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}"
            sentiment_response = self.session.get(sentiment_url, headers=headers)
            
            # Check response status for sentiment request
            if sentiment_response.status_code != 200:
                self.logger.error(f"Finnhub sentiment API returned status {sentiment_response.status_code} for {symbol}")
                sentiment_data = {"buzz": {}, "sentiment": {}}
            else:
                try:
                    sentiment_data = sentiment_response.json()
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to decode Finnhub sentiment response for {symbol}: {str(e)}")
                    sentiment_data = {"buzz": {}, "sentiment": {}}
            
            # Validate and sanitize news_data
            if news_data is None:
                self.logger.error(f"Null news data received for {symbol}")
                news_data = []
            elif not isinstance(news_data, list):
                self.logger.error(f"Invalid news data format for {symbol}: got {type(news_data)}")
                # Try to extract news array if it's nested in response
                if isinstance(news_data, dict):
                    news_data = news_data.get('data', []) or news_data.get('news', []) or []
                else:
                    news_data = []
            
            # Validate and sanitize sentiment_data
            if not isinstance(sentiment_data, dict):
                self.logger.error(f"Invalid sentiment data format for {symbol}: got {type(sentiment_data)}")
                sentiment_data = {"buzz": {}, "sentiment": {}}
            
            articles = []
            for item in news_data:
                if not isinstance(item, dict):
                    self.logger.warning(f"Skipping invalid news item for {symbol}: {type(item)}")
                    continue
                
                try:
                    # Get timestamp with fallback to current time
                    timestamp = item.get('datetime', 0)
                    if not timestamp:
                        self.logger.warning(f"Missing timestamp in news item for {symbol}")
                        continue
                    
                    # Ensure timezone-aware datetime
                    publish_time = datetime.fromtimestamp(timestamp).replace(tzinfo=pytz.UTC)
                    
                    # Changed condition and ensure both times are timezone-aware
                    if cutoff_time.replace(tzinfo=pytz.UTC) <= publish_time <= current_time.replace(tzinfo=pytz.UTC):
                        # Create article with sanitized data
                        article = {
                            "symbol": symbol,
                            "title": str(item.get('headline', '')).strip(),
                            "text": str(item.get('summary', '')).strip(),
                            "source": "Finnhub",
                            "published": publish_time.isoformat(),
                            "url": str(item.get('url', '')).strip(),
                            "sentiment": None
                        }
                        
                        # Only add article if it has either title or text
                        if article["title"] or article["text"]:
                            articles.append(article)
                        else:
                            self.logger.warning(f"Skipping empty article for {symbol}")
                
                except Exception as e:
                    self.logger.error(f"Error processing Finnhub news item for {symbol}: {str(e)}")
                    continue
            
            # Log summary of processed articles
            self.logger.info(f"Processed {len(articles)} articles for {symbol}")
            
            return {
                "type": "sentiment",
                "data": {
                    "symbol": symbol,
                    "buzz": sentiment_data.get('buzz', {}),
                    "sentiment": sentiment_data.get('sentiment', {}),
                    "articles": articles
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching Finnhub data for {symbol}: {str(e)}")
            return {"type": "sentiment", "data": {"symbol": symbol}}
    
    def _get_sec_filings(self, symbol: str, cutoff_time: datetime, current_time: datetime) -> Dict[str, Any]:
        """Get SEC filings from EDGAR"""
        try:
            # First, get the CIK number for the company
            cik = self._get_company_cik(symbol)
            if not cik:
                self.logger.warning(f"Could not find CIK for symbol: {symbol}")
                return {"type": "filing", "data": []}
            
            # Pad CIK with leading zeros to 10 digits as required by SEC
            cik_padded = str(cik).zfill(10)
            
            # Use SEC EDGAR API to get company filings
            url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
            headers = {
                'User-Agent': 'Multi-Agent-Quants research@example.com',  # Required by SEC
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'data.sec.gov'
            }
            
            response = self.session.get(url, headers=headers)
            if response.status_code != 200:
                self.logger.error(f"SEC API returned status code {response.status_code} for {symbol}")
                return {"type": "filing", "data": []}
            
            data = response.json()
            recent_filings = data.get('filings', {}).get('recent', {})
            
            if not recent_filings:
                self.logger.warning(f"No recent filings found for {symbol}")
                return {"type": "filing", "data": []}
            
            filings = []
            # Get the parallel arrays from the response
            form_array = recent_filings.get('form', [])
            date_array = recent_filings.get('filingDate', [])
            acc_num_array = recent_filings.get('accessionNumber', [])
            desc_array = recent_filings.get('primaryDocument', [])
            
            # Ensure cutoff_time and current_time are timezone-aware
            cutoff_time = cutoff_time.replace(tzinfo=pytz.UTC) if cutoff_time.tzinfo is None else cutoff_time
            current_time = current_time.replace(tzinfo=pytz.UTC) if current_time.tzinfo is None else current_time
            
            for i in range(min(len(form_array), 10)):  # Get last 10 filings
                try:
                    # Parse the filing date and make it timezone-aware
                    filing_date = datetime.strptime(date_array[i], '%Y-%m-%d').replace(tzinfo=pytz.UTC)
                    
                    # Compare timezone-aware datetimes
                    if cutoff_time <= filing_date <= current_time:
                        acc_num = acc_num_array[i].replace('-', '')
                        
                        filings.append({
                            "symbol": symbol,
                            "type": form_array[i],
                            "date": date_array[i],
                            "description": desc_array[i] if i < len(desc_array) else "",
                            "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num}/{desc_array[i]}"
                        })
                except Exception as e:
                    self.logger.error(f"Error processing filing {i} for {symbol}: {str(e)}")
                    continue
            
            return {"type": "filing", "data": filings}
            
        except Exception as e:
            self.logger.error(f"Error fetching SEC filings: {str(e)}")
            return {"type": "filing", "data": []}
    
    def _get_company_cik(self, symbol: str) -> str:
        """Get company CIK number from SEC"""
        try:
            # Try to get from cache first
            if hasattr(self, '_cik_cache') and symbol in self._cik_cache:
                return self._cik_cache[symbol]
            
            # Initialize cache if not exists
            if not hasattr(self, '_cik_cache'):
                self._cik_cache = {}
            
            # SEC company tickers JSON endpoint
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {
                'User-Agent': 'Multi-Agent-Quants research@example.com',
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'www.sec.gov'
            }
            
            response = self.session.get(url, headers=headers)
            if response.status_code != 200:
                self.logger.error(f"Failed to get company tickers from SEC: {response.status_code}")
                return None
            
            companies = response.json()
            
            # Search for the company by symbol
            for _, company in companies.items():
                if company['ticker'].upper() == symbol.upper():
                    cik = str(company['cik_str'])
                    self._cik_cache[symbol] = cik
                    return cik
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting CIK for {symbol}: {str(e)}")
            return None
    
    def _get_newsapi_articles(self, symbol: str, cutoff_time: datetime, current_time: datetime) -> Dict[str, Any]:
        """Get news from NewsAPI"""
        try:
            # Add date range to the query
            from_date = cutoff_time.strftime('%Y-%m-%d')
            to_date = current_time.strftime('%Y-%m-%d')
            
            url = (f"https://newsapi.org/v2/everything"
                   f"?q={symbol}"
                   f"&from={from_date}"
                   f"&to={to_date}"
                   f"&language=en"
                   f"&apiKey={self.newsapi_key}")
            
            response = self.session.get(url)
            data = response.json()
            
            articles = []
            for item in data.get('articles', []):
                publish_time = datetime.fromisoformat(item.get('publishedAt').replace('Z', '+00:00'))
                
                if publish_time >= cutoff_time:
                    continue
                
                articles.append({
                    "symbol": symbol,
                    "title": item.get('title'),
                    "text": item.get('description'),
                    "source": f"NewsAPI-{item.get('source', {}).get('name')}",
                    "published": publish_time.isoformat(),
                    "url": item.get('url'),
                    "sentiment": None
                })
            print(f"NewsAPI articles: {articles}")
            return {"type": "news", "data": articles}
            
        except Exception as e:
            self.logger.error(f"Error fetching NewsAPI articles: {str(e)}")
            return {"type": "news", "data": []}
    
    def _get_technical_indicators(self, symbols: List[str]) -> str:
        """Get technical indicators for symbols"""
        indicators = []
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1mo")
                
                # Calculate basic technical indicators
                sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                rsi = self._calculate_rsi(hist['Close'])
                
                indicators.append(
                    f"Symbol: {symbol}\n"
                    f"SMA20: {sma_20:.2f}\n"
                    f"SMA50: {sma_50:.2f}\n"
                    f"RSI: {rsi:.2f}\n"
                    f"Volume Trend: {'Up' if hist['Volume'].iloc[-1] > hist['Volume'].mean() else 'Down'}"
                )
            except Exception as e:
                self.logger.error(f"Error calculating technical indicators: {str(e)}")
                continue
        
        return "\n\n".join(indicators)
    
    def _calculate_rsi(self, prices, periods=14):
        """Calculate RSI technical indicator"""
        deltas = prices.diff()
        gain = (deltas.where(deltas > 0, 0)).rolling(window=periods).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))
    
    def _format_articles(self, articles: List[Dict[str, Any]]) -> str:
        """Format articles for analysis"""
        # Sort by publish time
        articles.sort(key=lambda x: x['published'], reverse=True)
        
        return "\n\n".join([
            f"Symbol: {article['symbol']}\n"
            f"Title: {article['title']}\n"
            f"Source: {article['source']}\n"
            f"Published: {article['published']}\n"
            f"Summary: {article['text']}\n"
            f"Sentiment: {article.get('sentiment', 'N/A')}"
            for article in articles
        ])
    
    def _format_sentiment(self, sentiment_data: List[Dict[str, Any]]) -> str:
        """Format sentiment data for analysis"""
        formatted = []
        for data in sentiment_data:
            if not data:
                continue
            
            symbol = data.get('symbol')
            buzz = data.get('buzz', {})
            sentiment = data.get('sentiment', {})
            
            formatted.append(
                f"Symbol: {symbol}\n"
                f"Article Mentions: {buzz.get('articlesInLastWeek', 'N/A')}\n"
                f"Buzz Score: {buzz.get('buzz', 'N/A')}\n"
                f"Sentiment Score: {sentiment.get('score', 'N/A')}\n"
                f"Bullish Percent: {sentiment.get('bullishPercent', 'N/A')}%\n"
                f"Bearish Percent: {sentiment.get('bearishPercent', 'N/A')}%"
            )
        
        return "\n\n".join(formatted)
    
    def _format_filings(self, filings: List[Dict[str, Any]]) -> str:
        """Format SEC filings for analysis"""
        # Sort by date
        filings.sort(key=lambda x: x['date'], reverse=True)
        
        return "\n\n".join([
            f"Symbol: {filing['symbol']}\n"
            f"Type: {filing['type']}\n"
            f"Date: {filing['date']}\n"
            f"Description: {filing['description']}\n"
            f"URL: {filing['url']}"
            for filing in filings
        ])
    
    def _calculate_sentiment_scores(self, articles: List[Dict[str, Any]], 
                                 sentiment_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregated sentiment scores"""
        scores = {}
        for article in articles:
            symbol = article['symbol']
            if symbol not in scores:
                scores[symbol] = []
            
            if article.get('sentiment') is not None:
                scores[symbol].append(float(article['sentiment']))
        
        # Add Finnhub sentiment
        for data in sentiment_data:
            if not data:
                continue
            
            symbol = data.get('symbol')
            sentiment = data.get('sentiment', {})
            if sentiment.get('score') is not None:
                if symbol not in scores:
                    scores[symbol] = []
                scores[symbol].append(float(sentiment['score']))
        
        # Calculate average scores
        return {
            symbol: sum(scores) / len(scores) if scores else 0.0
            for symbol, scores in scores.items()
        }
    
    def _calculate_source_breakdown(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate breakdown of news sources"""
        sources = {}
        for article in articles:
            source = article['source']
            sources[source] = sources.get(source, 0) + 1
        return sources
    
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
    
    def _calculate_confidence_score(self, news_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on news data quality and quantity.
        
        Args:
            news_data: Dictionary containing news articles, sentiment, and source information
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            # Initialize scoring factors
            factors = {
                'source_diversity': 0.0,  # Higher score for multiple sources
                'article_quantity': 0.0,  # Higher score for more articles
                'source_quality': 0.0,    # Higher score for reliable sources
                'data_freshness': 0.0,    # Higher score for recent data
            }
            
            # 1. Source Diversity
            unique_sources = set()
            for article in news_data.get('articles', '').split('\n\n'):
                if 'Source:' in article:
                    source = article.split('Source:')[1].split('\n')[0].strip()
                    unique_sources.add(source)
            
            factors['source_diversity'] = min(len(unique_sources) / 5, 1.0)  # Normalize to max of 5 sources
            
            # 2. Article Quantity
            article_count = len(news_data.get('articles', '').split('\n\n'))
            factors['article_quantity'] = min(article_count / 10, 1.0)  # Normalize to max of 10 articles
            
            # 3. Source Quality
            high_quality_sources = {
                'SEC', 'Alpha Vantage', 'YFinance', 'Bloomberg', 'Reuters', 
                'Financial Times', 'Wall Street Journal', 'EDGAR'
            }
            quality_sources = sum(1 for source in unique_sources if any(qs in source for qs in high_quality_sources))
            factors['source_quality'] = quality_sources / max(len(unique_sources), 1)
            
            # 4. Data Freshness
            current_time = datetime.now(pytz.UTC)
            timestamps = []
            for article in news_data.get('articles', '').split('\n\n'):
                if 'Published:' in article:
                    try:
                        published_str = article.split('Published:')[1].split('\n')[0].strip()
                        published_time = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                        timestamps.append(published_time)
                    except (ValueError, IndexError):
                        continue
            
            if timestamps:
                newest_article = max(timestamps)
                hours_old = (current_time - newest_article).total_seconds() / 3600
                factors['data_freshness'] = max(0, 1 - (hours_old / 24))  # Normalize to 24 hours
            
            # # 5. Sentiment Strength
            # sentiment_scores = news_data.get('sentiment_scores', {})
            # if sentiment_scores:
            #     # Calculate average absolute deviation from neutral (0.5)
            #     deviations = [abs(score - 0.5) * 2 for score in sentiment_scores.values()]
            #     factors['sentiment_strength'] = sum(deviations) / len(deviations)
            
            # Calculate weighted average of factors
            weights = {
                'source_diversity': 0.2,
                'article_quantity': 0.15,
                'source_quality': 0.3,
                'data_freshness': 0.2,
            }
            
            confidence_score = sum(score * weights[factor] for factor, score in factors.items())
            
            # Log the factors for debugging
            self.logger.debug(f"Confidence factors: {factors}")
            self.logger.debug(f"Final confidence score: {confidence_score}")
            
            return confidence_score
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5  # Return neutral confidence on error

