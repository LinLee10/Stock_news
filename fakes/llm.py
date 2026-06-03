#!/usr/bin/env python3
"""Fake LLM clients for DRY_RUN mode"""
from typing import Dict, List, Any, Optional
import json
import hashlib


class FakeLLMClient:
    """Fake LLM client that returns deterministic responses"""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.call_count = 0
    
    def _generate_response(self, prompt: str) -> str:
        """Generate deterministic response based on prompt"""
        # Create deterministic response based on prompt hash
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        
        if "analysis" in prompt.lower():
            responses = [
                "Based on the data, I observe positive trends in the market.",
                "The analysis suggests moderate volatility with upward momentum.", 
                "Technical indicators show consolidation before potential breakout.",
                "Market sentiment appears cautiously optimistic."
            ]
        elif "summary" in prompt.lower():
            responses = [
                "Key points: Market volatility, tech sector strength, economic indicators positive.",
                "Summary: Trading volume increased, prices stabilized, outlook remains positive.",
                "Main highlights: Strong earnings, sector rotation, consumer confidence up.",
                "Overview: Mixed signals but generally positive momentum observed."
            ]
        elif "recommendation" in prompt.lower():
            responses = [
                "Recommend holding current positions with careful monitoring.",
                "Consider increasing allocation to growth stocks gradually.",
                "Maintain diversified portfolio with focus on quality names.",
                "Suggest defensive positioning given current uncertainty."
            ]
        else:
            responses = [
                "The market analysis indicates balanced conditions.",
                "Current trends suggest stable performance ahead.",
                "Technical patterns show consolidation phase.",
                "Fundamentals remain supportive for equities."
            ]
        
        return responses[prompt_hash % len(responses)]
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Fake chat completion"""
        self.call_count += 1
        
        # Get last user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break
        
        response_text = self._generate_response(user_msg)
        
        return {
            "id": f"fake-{self.provider}-{self.call_count}",
            "object": "chat.completion",
            "created": 1705320000,
            "model": f"fake-{self.provider}-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_msg) // 4,  # Rough estimate
                "completion_tokens": len(response_text) // 4,
                "total_tokens": (len(user_msg) + len(response_text)) // 4
            }
        }
    
    def chat_completion_sync(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Synchronous version"""
        import asyncio
        return asyncio.run(self.chat_completion(messages, **kwargs))
    
    async def completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Fake text completion"""
        self.call_count += 1
        
        response_text = self._generate_response(prompt)
        
        return {
            "id": f"fake-{self.provider}-{self.call_count}",
            "object": "text_completion", 
            "created": 1705320000,
            "model": f"fake-{self.provider}-model",
            "choices": [{
                "text": response_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(response_text) // 4,
                "total_tokens": (len(prompt) + len(response_text)) // 4
            }
        }


class FakeOpenAIClient(FakeLLMClient):
    """Fake OpenAI client"""
    
    def __init__(self, api_key: str = "fake-key"):
        super().__init__("openai")
        self.api_key = api_key
    
    class ChatCompletion:
        def __init__(self, client):
            self.client = client
        
        async def create(self, messages: List[Dict], **kwargs):
            return await self.client.chat_completion(messages, **kwargs)
        
        def create_sync(self, messages: List[Dict], **kwargs):
            return self.client.chat_completion_sync(messages, **kwargs)
    
    class Completion:
        def __init__(self, client):
            self.client = client
        
        async def create(self, prompt: str, **kwargs):
            return await self.client.completion(prompt, **kwargs)
    
    def __init__(self, api_key: str = "fake-key"):
        super().__init__("openai")
        self.api_key = api_key
        self.chat = self.ChatCompletion(self)
        self.completions = self.Completion(self)


class FakeAnthropicClient(FakeLLMClient):
    """Fake Anthropic client"""
    
    def __init__(self, api_key: str = "fake-key"):
        super().__init__("anthropic")
        self.api_key = api_key
    
    class Messages:
        def __init__(self, client):
            self.client = client
            
        async def create(self, messages: List[Dict], **kwargs):
            # Convert to Anthropic format
            response = await self.client.chat_completion(messages, **kwargs)
            return {
                "id": response["id"],
                "type": "message",
                "role": "assistant", 
                "content": [{"type": "text", "text": response["choices"][0]["message"]["content"]}],
                "model": response["model"],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": response["usage"]["prompt_tokens"],
                    "output_tokens": response["usage"]["completion_tokens"]
                }
            }
    
    def __init__(self, api_key: str = "fake-key"):
        super().__init__("anthropic")
        self.api_key = api_key
        self.messages = self.Messages(self)


# Factory functions
def create_openai_client(api_key: str = None) -> FakeOpenAIClient:
    """Create fake OpenAI client"""
    return FakeOpenAIClient(api_key or "fake-openai-key")


def create_anthropic_client(api_key: str = None) -> FakeAnthropicClient:
    """Create fake Anthropic client"""
    return FakeAnthropicClient(api_key or "fake-anthropic-key")