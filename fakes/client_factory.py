#!/usr/bin/env python3
"""Fake client factory for DRY_RUN mode"""
from typing import Any, Optional


class FakeClientFactory:
    """Factory that returns fake clients for all external services"""
    
    def get_alpha_vantage_client(self, api_key: Optional[str] = None) -> Any:
        """Return fake Alpha Vantage client"""
        from fakes.alpha_vantage import FakeAlphaVantageManager
        return FakeAlphaVantageManager(api_key or "FAKE_KEY")
    
    def get_email_client(self) -> Any:
        """Return fake email client"""
        from fakes.email import FakeEmailSender
        return FakeEmailSender()
    
    def get_llm_client(self, provider: str = "openai") -> Any:
        """Return fake LLM client"""
        return FakeLLMClient(provider)
    
    def get_news_client(self) -> Any:
        """Return fake news client"""
        return FakeNewsClient()
    
    def get_database_client(self) -> Any:
        """Return fake/in-memory database client"""
        return FakeDatabaseClient()


class FakeLLMClient:
    """Fake LLM client for OpenAI/Anthropic/etc"""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.call_count = 0
    
    async def complete(self, prompt: str, **kwargs) -> dict:
        """Return fake completion"""
        self.call_count += 1
        return {
            "choices": [{
                "message": {
                    "content": f"Fake {self.provider} response to: {prompt[:50]}..."
                }
            }]
        }
    
    async def chat_completion(self, messages: list, **kwargs) -> dict:
        """Return fake chat completion"""
        self.call_count += 1
        last_message = messages[-1].get("content", "") if messages else ""
        return {
            "choices": [{
                "message": {
                    "content": f"Fake {self.provider} chat response to: {last_message[:50]}..."
                }
            }]
        }


class FakeNewsClient:
    """Fake news client"""
    
    def __init__(self):
        self.call_count = 0
    
    async def get_news(self, query: str = "", **kwargs) -> dict:
        """Return fake news data"""
        self.call_count += 1
        return {
            "articles": [
                {
                    "title": f"Fake news article about {query}",
                    "description": "This is fake news for testing",
                    "url": "https://example.com/fake-news",
                    "publishedAt": "2024-01-15T10:00:00Z",
                    "source": {"name": "Fake News Source"}
                }
            ]
        }


class FakeDatabaseClient:
    """Fake in-memory database client"""
    
    def __init__(self):
        self._data = {}
        self.call_count = 0
    
    async def execute(self, query: str, params: list = None) -> dict:
        """Execute fake query"""
        self.call_count += 1
        return {"rows_affected": 1, "data": []}
    
    async def fetch(self, query: str, params: list = None) -> list:
        """Fetch fake data"""
        self.call_count += 1
        return []
    
    async def insert(self, table: str, data: dict) -> dict:
        """Insert fake data"""
        self.call_count += 1
        if table not in self._data:
            self._data[table] = []
        self._data[table].append(data)
        return {"id": len(self._data[table])}
    
    async def close(self):
        """Close fake connection"""
        pass