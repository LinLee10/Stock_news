#!/usr/bin/env python3
"""
Usage examples for the secrets management system
Shows how to integrate API keys into your services
"""

from load_secrets import SecretsManager, get_financial_api_key, get_news_api_key


def example_news_service():
    """Example of using news API keys"""
    print("📰 News Service Examples")
    print("-" * 30)
    
    # NewsAPI example
    newsapi_key = get_news_api_key('newsapi', 'production')
    if newsapi_key:
        print(f"NewsAPI: Ready with key ending in ...{newsapi_key[-4:]}")
        
        # Example usage in requests
        import requests
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'tesla stock',
            'apiKey': newsapi_key,
            'language': 'en',
            'sortBy': 'publishedAt'
        }
        # response = requests.get(url, params=params)
        print(f"Would call: {url} with key")
    
    # GNews example  
    gnews_key = get_news_api_key('gnews', 'production')
    if gnews_key:
        print(f"GNews: Ready with key ending in ...{gnews_key[-4:]}")


def example_financial_service():
    """Example of using financial data API keys"""
    print("\n💰 Financial Service Examples")
    print("-" * 30)
    
    # Finnhub example
    finnhub_key = get_financial_api_key('finnhub', 'production')
    if finnhub_key:
        print(f"Finnhub: Ready with key ending in ...{finnhub_key[-4:]}")
        
        # Example WebSocket connection
        websocket_url = f"wss://ws.finnhub.io?token={finnhub_key}"
        print(f"WebSocket URL ready: {websocket_url[:50]}...")
    
    # Polygon example
    polygon_key = get_financial_api_key('polygon', 'production')  
    if polygon_key:
        print(f"Polygon: Ready with key ending in ...{polygon_key[-4:]}")
        
        # Example REST API call
        api_url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09?apikey={polygon_key}"
        print(f"API URL ready: {api_url[:80]}...")


def example_government_data():
    """Example of using government data API keys"""  
    print("\n🏛️  Government Data Examples")
    print("-" * 30)
    
    from load_secrets import get_government_api_key
    
    # FRED (Federal Reserve) example
    fred_key = get_government_api_key('fred', 'production')
    if fred_key:
        print(f"FRED: Ready with key ending in ...{fred_key[-4:]}")
        
        # Example FRED API call for unemployment rate
        fred_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={fred_key}&file_type=json"
        print(f"FRED URL ready for unemployment data")


def example_alpaca_trading():
    """Example of using Alpaca trading credentials"""
    print("\n📈 Alpaca Trading Example") 
    print("-" * 30)
    
    from load_secrets import get_alpaca_credentials
    
    alpaca_creds = get_alpaca_credentials('production')
    if alpaca_creds['api_key'] and alpaca_creds['secret_key']:
        print(f"Alpaca API Key: ...{alpaca_creds['api_key'][-4:]}")
        print(f"Alpaca Secret: {'*' * 20}")
        
        # Example Alpaca API client setup
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(
                alpaca_creds['api_key'],
                alpaca_creds['secret_key'], 
                base_url='https://paper-api.alpaca.markets'  # Paper trading
            )
            print("Alpaca client ready for paper trading")
        except ImportError:
            print("(alpaca_trade_api not installed - pip install alpaca-trade-api)")


def example_service_integration():
    """Example of integrating secrets into a service class"""
    print("\n🔧 Service Integration Example")
    print("-" * 30)
    
    class StockDataService:
        def __init__(self, environment='production'):
            self.secrets = SecretsManager(environment)
            self.finnhub_key = self.secrets.get_financial_api_key('finnhub')
            self.polygon_key = self.secrets.get_financial_api_key('polygon') 
            self.news_key = self.secrets.get_news_api_key('newsapi')
            
        def get_stock_quote(self, symbol):
            if not self.finnhub_key:
                raise ValueError("Finnhub API key not configured")
            # Implementation here
            return f"Would fetch {symbol} quote using Finnhub"
            
        def get_stock_news(self, symbol):
            if not self.news_key:
                raise ValueError("News API key not configured") 
            # Implementation here
            return f"Would fetch {symbol} news using NewsAPI"
    
    # Usage
    service = StockDataService('production')
    print(service.get_stock_quote('AAPL'))
    print(service.get_stock_news('AAPL'))


if __name__ == "__main__":
    print("🔐 API Keys Usage Examples")
    print("=" * 50)
    
    try:
        example_news_service()
        example_financial_service()
        example_government_data()
        example_alpaca_trading()
        example_service_integration()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        print("Make sure your secrets are properly configured.")