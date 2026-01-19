"""
GOOGLE GEMINI 2.5 PRO PROVIDER FOR SADIE AI
=============================================
Primary LLM provider using Google's most advanced model.

Features:
- Gemini 2.5 Pro as primary model
- 1M token context window
- Advanced reasoning capabilities
- Native JSON mode support
- Automatic fallback handling
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List


class GeminiProvider:
    """
    Google Gemini 2.5 Pro LLM provider.
    Primary model for Sadie AI with superior reasoning.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "gemini-2.5-pro-preview-06-05"  # Latest Gemini 2.5 Pro
        self.fallback_model = "gemini-2.5-flash-preview-05-20"  # Fast fallback
        
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using Gemini 2.5 Pro.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
            system_instruction: Optional system prompt
            
        Returns:
            Dict with 'success', 'content', 'model', 'usage'
        """
        result = {
            "success": False,
            "content": "",
            "model": self.model,
            "usage": {},
            "error": None
        }
        
        if not self.api_key:
            result["error"] = "GEMINI_API_KEY not set"
            return result
        
        # Convert messages to Gemini format
        contents = []
        system_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Gemini uses 'user' and 'model' roles
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
            else:
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
        
        # Combine system instructions
        if system_instruction:
            system_parts.insert(0, system_instruction)
        
        combined_system = "\n\n".join(system_parts) if system_parts else None
        
        # Build request payload
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        if combined_system:
            payload["systemInstruction"] = {
                "parts": [{"text": combined_system}]
            }
        
        # Try primary model first
        for model in [self.model, self.fallback_model]:
            try:
                url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
                
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=180  # 3 min timeout for complex queries
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract response content
                    candidates = data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            result["content"] = parts[0].get("text", "")
                            result["success"] = True
                            result["model"] = model
                            
                            # Extract usage if available
                            usage = data.get("usageMetadata", {})
                            result["usage"] = {
                                "prompt_tokens": usage.get("promptTokenCount", 0),
                                "completion_tokens": usage.get("candidatesTokenCount", 0),
                                "total_tokens": usage.get("totalTokenCount", 0)
                            }
                            
                            return result
                else:
                    print(f"Gemini {model} error: {response.status_code} - {response.text[:200]}")
                    
            except Exception as e:
                print(f"Gemini {model} exception: {e}")
                continue
        
        result["error"] = "All Gemini models failed"
        return result
    
    def generate_with_thinking(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 16384,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate with extended thinking for complex analysis.
        Uses higher token limits and structured reasoning.
        """
        # Add thinking prompt
        thinking_instruction = """
You are performing deep financial analysis. Think through this step-by-step:

1. First, analyze all the data provided carefully
2. Identify key patterns, signals, and anomalies
3. Consider both bullish and bearish scenarios
4. Weigh the evidence and assign confidence levels
5. Formulate your recommendation with clear reasoning

Show your reasoning process, then provide your final analysis.
"""
        
        combined_system = f"{system_instruction}\n\n{thinking_instruction}" if system_instruction else thinking_instruction
        
        return self.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            system_instruction=combined_system
        )


class PerplexityProvider:
    """
    Perplexity Sonar Pro for real-time research.
    Complementary to Gemini for current events and news.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('PERPLEXITY_API_KEY', '')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar-pro"
        
    def research(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Perform real-time research using Perplexity Sonar Pro.
        
        Args:
            query: Research query
            context: Optional context to include
            max_tokens: Maximum response tokens
            
        Returns:
            Dict with research results and citations
        """
        result = {
            "success": False,
            "content": "",
            "citations": [],
            "error": None
        }
        
        if not self.api_key:
            result["error"] = "PERPLEXITY_API_KEY not set"
            return result
        
        messages = []
        
        if context:
            messages.append({
                "role": "system",
                "content": f"Context for research:\n{context}"
            })
        
        messages.append({
            "role": "user",
            "content": query
        })
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,  # Lower temp for factual research
            "return_citations": True
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                choices = data.get("choices", [])
                if choices:
                    result["content"] = choices[0].get("message", {}).get("content", "")
                    result["success"] = True
                
                result["citations"] = data.get("citations", [])
                
            else:
                result["error"] = f"Perplexity error: {response.status_code}"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result


class MultiModelOrchestrator:
    """
    Orchestrates multiple LLM providers for optimal results.
    
    Strategy:
    1. Gemini 2.5 Pro for main analysis (primary)
    2. Perplexity Sonar Pro for real-time research (complementary)
    3. OpenRouter fallback for reliability
    """
    
    def __init__(self):
        self.gemini = GeminiProvider()
        self.perplexity = PerplexityProvider()
        self.openrouter_key = os.environ.get('OPENROUTER_API_KEY', '')
        
    def analyze(
        self,
        user_message: str,
        context: str,
        system_prompt: str,
        is_nuke_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using multiple models.
        
        Args:
            user_message: User's query
            context: Market data context
            system_prompt: System instructions
            is_nuke_mode: Whether to use extended analysis
            
        Returns:
            Dict with analysis results
        """
        result = {
            "success": False,
            "content": "",
            "model_used": "",
            "perplexity_research": None,
            "error": None
        }
        
        # Step 1: Get real-time research from Perplexity
        if self.perplexity.api_key:
            research_query = f"Latest news, analyst opinions, and market sentiment for: {user_message}"
            perplexity_result = self.perplexity.research(research_query)
            
            if perplexity_result.get("success"):
                result["perplexity_research"] = perplexity_result.get("content")
        
        # Step 2: Build messages for Gemini
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"REAL-TIME MARKET DATA:\n{context}"}
        ]
        
        # Add Perplexity research if available
        if result["perplexity_research"]:
            messages.append({
                "role": "system",
                "content": f"PERPLEXITY REAL-TIME RESEARCH:\n{result['perplexity_research']}"
            })
        
        messages.append({"role": "user", "content": user_message})
        
        # Step 3: Generate with Gemini
        max_tokens = 16384 if is_nuke_mode else 8192
        
        if is_nuke_mode:
            gemini_result = self.gemini.generate_with_thinking(
                messages=messages,
                max_tokens=max_tokens,
                system_instruction=system_prompt
            )
        else:
            gemini_result = self.gemini.generate(
                messages=messages,
                max_tokens=max_tokens,
                system_instruction=system_prompt
            )
        
        if gemini_result.get("success"):
            result["success"] = True
            result["content"] = gemini_result.get("content", "")
            result["model_used"] = f"gemini/{gemini_result.get('model', 'unknown')}"
            result["usage"] = gemini_result.get("usage", {})
            return result
        
        # Step 4: Fallback to OpenRouter if Gemini fails
        if self.openrouter_key:
            fallback_result = self._openrouter_fallback(messages, max_tokens)
            if fallback_result.get("success"):
                result.update(fallback_result)
                return result
        
        result["error"] = "All models failed"
        return result
    
    def _openrouter_fallback(
        self,
        messages: List[Dict],
        max_tokens: int
    ) -> Dict[str, Any]:
        """Fallback to OpenRouter models."""
        result = {
            "success": False,
            "content": "",
            "model_used": ""
        }
        
        fallback_models = [
            "google/gemini-2.5-pro-preview",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o"
        ]
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json"
        }
        
        for model in fallback_models:
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    if content:
                        result["success"] = True
                        result["content"] = content
                        result["model_used"] = model
                        return result
                        
            except Exception as e:
                print(f"OpenRouter fallback {model} failed: {e}")
                continue
        
        return result


# Singleton instances
_gemini_instance = None
_orchestrator_instance = None

def get_gemini_provider() -> GeminiProvider:
    """Get or create Gemini provider instance."""
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = GeminiProvider()
    return _gemini_instance

def get_multi_model_orchestrator() -> MultiModelOrchestrator:
    """Get or create multi-model orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = MultiModelOrchestrator()
    return _orchestrator_instance


if __name__ == "__main__":
    # Test the providers
    print("Testing Gemini Provider...")
    gemini = GeminiProvider()
    
    result = gemini.generate(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=100
    )
    
    print(f"Success: {result['success']}")
    print(f"Model: {result['model']}")
    print(f"Content: {result['content'][:200]}")
