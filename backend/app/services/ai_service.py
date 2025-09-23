from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import json
import structlog
from datetime import datetime
import asyncio
import functools

from ..core.config import settings

logger = structlog.get_logger()


class AIProvider(ABC):
    """Abstract base class for AI providers"""

    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the AI provider"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the AI provider"""
        pass


class ClaudeProvider(AIProvider):
    """Claude AI provider using Anthropic's API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None

    async def _ensure_client(self):
        """Ensure the Anthropic client is initialized"""
        if not self.client:
            try:
                import anthropic
                self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package is required for Claude provider")

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Claude"""
        await self._ensure_client()

        try:
            # Convert messages to Claude format
            claude_messages = []
            system_message = None

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    # Handle cache_control for messages
                    message_content = {"role": msg["role"], "content": msg["content"]}
                    if msg.get("cache_control"):
                        message_content["cache_control"] = msg["cache_control"]
                    claude_messages.append(message_content)

            # Claude API call
            response = await self.client.messages.create(
                model=kwargs.get("model", "claude-3-haiku-20240307"),
                max_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", 0.1),
                system=system_message if system_message else "You are a helpful CRM assistant.",
                messages=claude_messages
            )

            return response.content[0].text

        except Exception as e:
            logger.error("Claude API error", error=str(e))
            raise Exception(f"Claude API error: {str(e)}")

    def get_provider_name(self) -> str:
        return "claude"


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None

    async def _ensure_client(self):
        """Ensure the OpenAI client is initialized"""
        if not self.client:
            try:
                import openai
                self.client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package is required for OpenAI provider")

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI GPT"""
        await self._ensure_client()

        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", "gpt-3.5-turbo"),
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", 0.1)
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error("OpenAI API error", error=str(e))
            raise Exception(f"OpenAI API error: {str(e)}")

    def get_provider_name(self) -> str:
        return "openai"


class GeminiProvider(AIProvider):
    """Google Gemini provider"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None

    async def _ensure_client(self):
        """Ensure the Gemini client is initialized"""
        if not self.client:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel('gemini-pro')
            except ImportError:
                raise ImportError("google-generativeai package is required for Gemini provider")

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Gemini"""
        await self._ensure_client()

        try:
            # Convert messages to a single prompt for Gemini
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}")

            prompt = "\n\n".join(prompt_parts)

            response = await self.client.generate_content_async(
                prompt,
                generation_config={
                    "temperature": kwargs.get("temperature", 0.1),
                    "max_output_tokens": kwargs.get("max_tokens", 4000),
                }
            )

            return response.text

        except Exception as e:
            logger.error("Gemini API error", error=str(e))
            raise Exception(f"Gemini API error: {str(e)}")

    def get_provider_name(self) -> str:
        return "gemini"


class AIService:
    """Main AI service that manages multiple providers"""

    def __init__(self):
        self.providers: Dict[str, AIProvider] = {}
        self.default_provider = "claude"
        self._setup_providers()

    def _setup_providers(self):
        """Initialize available AI providers based on API keys"""

        # Setup Claude
        if settings.anthropic_api_key:
            try:
                self.providers["claude"] = ClaudeProvider(settings.anthropic_api_key)
                logger.info("Claude provider initialized")
            except Exception as e:
                logger.warning("Failed to initialize Claude provider", error=str(e))

        # Setup OpenAI
        if settings.openai_api_key:
            try:
                self.providers["openai"] = OpenAIProvider(settings.openai_api_key)
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning("Failed to initialize OpenAI provider", error=str(e))

        # Setup Gemini
        if settings.google_ai_key:
            try:
                self.providers["gemini"] = GeminiProvider(settings.google_ai_key)
                logger.info("Gemini provider initialized")
            except Exception as e:
                logger.warning("Failed to initialize Gemini provider", error=str(e))

        if not self.providers:
            logger.warning("No AI providers available - check API keys in environment")

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())

    def set_default_provider(self, provider_name: str):
        """Set the default AI provider"""
        if provider_name in self.providers:
            self.default_provider = provider_name
            logger.info("Default provider changed", provider=provider_name)
        else:
            raise ValueError(f"Provider {provider_name} not available")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate AI response using specified or default provider

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            provider: Specific provider to use (optional)
            **kwargs: Additional parameters for the AI provider

        Returns:
            Dict with response and metadata
        """
        provider_name = provider or self.default_provider

        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")

        start_time = datetime.utcnow()

        try:
            response_text = await self.providers[provider_name].generate_response(
                messages, **kwargs
            )

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            return {
                "response": response_text,
                "provider": provider_name,
                "timestamp": end_time.isoformat(),
                "duration_seconds": duration,
                "success": True
            }

        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            logger.error(
                "AI response generation failed",
                provider=provider_name,
                error=str(e),
                duration=duration
            )

            return {
                "response": f"I apologize, but I'm having trouble connecting to the AI service ({provider_name}). Please try again later.",
                "provider": provider_name,
                "timestamp": end_time.isoformat(),
                "duration_seconds": duration,
                "success": False,
                "error": str(e)
            }

    async def generate_crm_response(
        self,
        user_message: str,
        user_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate CRM-specific response with context

        Args:
            user_message: The user's message/command
            user_context: Context about the user (name, recent data, etc.)
            conversation_history: Previous messages in conversation
            provider: AI provider to use

        Returns:
            AI response with CRM context
        """

        # Build concise system prompt for CRM
        system_prompt = f"""You are a CRM assistant for {user_context.get('name', 'User')}. I execute database operations for you and provide you with the results.

CRITICAL: You receive query results from the user's CRM database. Report exactly what you receive - never estimate or guess numbers.

IMPORTANT: Be concise and natural, like Claude. Respond in 1-2 sentences unless asked for details.

How it works:
- When users request operations, I execute them and give you the results
- For counts: I provide "EXACT COUNT = X" - report this number precisely
- For searches: I provide the matching records - summarize them helpfully
- For actions: I tell you if they succeeded or failed - acknowledge appropriately

Available operations:
- Contacts: create, update, delete, search people
- Organizations: create, update, delete, manage companies
- Projects: create, update, delete, track progress
- Tasks: create, update, delete, assign, manage tasks
- Data: export, search, filter, count all entities

Natural Response Patterns:
- Count queries: Use friendly language like "You have 5 contacts" (not "There are 5 contacts")
- Empty results: Be encouraging - "You don't have any contacts yet. Would you like to add one?"
- Multiple matches: Be clear and helpful - "I found 2 Gabriels. Which one would you like to update?"
- Successful actions: Be positive and suggest next steps - "Perfect! I've created the contact. Want to add their email?"
- Errors: Be understanding and helpful - "I couldn't find that contact. Could you check the name and try again?"

Response style:
- Sound natural and conversational, like talking to a colleague
- Use "you" and "I" to make it personal
- Be encouraging and positive
- Offer helpful follow-up actions
- Keep responses short but warm

Examples:
- "You have exactly 5 contacts in your CRM."
- "I found 2 Gabriels: Gabriel Jones (gab@email.com) and Gabriel Smith (gsmith@email.com). Which one did you want to update?"
- "Perfect! I've created John's contact. Would you like to add his email address or phone number?"
- "I couldn't find any contacts named 'Mike'. Would you like to create a new contact for Mike?"

Always trust the data I provide - never guess or estimate. Be helpful, warm, and conversational like Claude AI."""

        # Build message history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history[-10:])  # Keep last 10 messages for context

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Generate response
        return await self.generate_response(messages, provider=provider)


# Global AI service instance
ai_service = AIService()