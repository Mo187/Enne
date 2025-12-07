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
    """Claude AI provider using Anthropic's API with extended thinking support"""

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

    async def generate_response_with_thinking(
        self,
        messages: List[Dict[str, str]],
        thinking_budget: int = 10000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using Claude with extended thinking mode.

        Extended thinking enables deeper reasoning for complex tasks like:
        - Ambiguous intent resolution
        - Multi-step planning
        - Complex entity disambiguation

        Args:
            messages: Conversation messages
            thinking_budget: Max tokens for thinking (default 10000)
            **kwargs: Additional parameters

        Returns:
            Dict with 'response', 'thinking' (optional), and 'used_thinking'
        """
        await self._ensure_client()

        try:
            # Convert messages to Claude format
            claude_messages = []
            system_message = None

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    message_content = {"role": msg["role"], "content": msg["content"]}
                    if msg.get("cache_control"):
                        message_content["cache_control"] = msg["cache_control"]
                    claude_messages.append(message_content)

            # Use a model that supports extended thinking
            # claude-sonnet-4-20250514 or claude-3-5-sonnet-20241022
            thinking_model = kwargs.get("model", "claude-sonnet-4-20250514")

            logger.info(
                "Using extended thinking mode",
                model=thinking_model,
                thinking_budget=thinking_budget
            )

            # Claude API call with extended thinking
            response = await self.client.messages.create(
                model=thinking_model,
                max_tokens=16000,  # Extended for thinking + response
                temperature=1.0,  # Required for extended thinking
                thinking={
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                },
                system=system_message if system_message else "You are a helpful CRM assistant.",
                messages=claude_messages
            )

            # Extract thinking and response from content blocks
            thinking_content = ""
            response_content = ""

            for block in response.content:
                if block.type == "thinking":
                    thinking_content = block.thinking
                elif block.type == "text":
                    response_content = block.text

            logger.info(
                "Extended thinking completed",
                thinking_length=len(thinking_content),
                response_length=len(response_content)
            )

            return {
                "response": response_content,
                "thinking": thinking_content,
                "used_thinking": True,
                "model": thinking_model
            }

        except TypeError as e:
            # SDK version doesn't support 'thinking' parameter
            if "thinking" in str(e):
                logger.warning(
                    "Extended thinking not supported by current Anthropic SDK version. "
                    "Upgrade anthropic package to 0.49.0+ for extended thinking support.",
                    error=str(e)
                )
            else:
                logger.error("Claude extended thinking TypeError", error=str(e))
            # Fall back to regular generation
            regular_response = await self.generate_response(messages, **kwargs)
            return {
                "response": regular_response,
                "thinking": None,
                "used_thinking": False,
                "fallback": True,
                "fallback_reason": "SDK does not support extended thinking"
            }
        except Exception as e:
            logger.error("Claude extended thinking error", error=str(e))
            # Fall back to regular generation
            logger.info("Falling back to regular response generation")
            regular_response = await self.generate_response(messages, **kwargs)
            return {
                "response": regular_response,
                "thinking": None,
                "used_thinking": False,
                "fallback": True,
                "fallback_reason": str(e)
            }

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

    async def generate_response_with_thinking(
        self,
        messages: List[Dict[str, str]],
        thinking_budget: int = 10000,
        provider: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate AI response with extended thinking mode (Claude only).

        Extended thinking enables deeper reasoning for complex tasks like:
        - Ambiguous intent resolution
        - Multi-step planning
        - Complex entity disambiguation
        - Low confidence situations

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            thinking_budget: Max tokens for thinking (default 10000)
            provider: Specific provider to use (optional, defaults to Claude)
            **kwargs: Additional parameters for the AI provider

        Returns:
            Dict with response, thinking content (if available), and metadata
        """
        provider_name = provider or self.default_provider

        # Extended thinking only works with Claude
        if provider_name != "claude" or "claude" not in self.providers:
            logger.info("Extended thinking requested but provider is not Claude, using regular generation")
            return await self.generate_response(messages, provider=provider_name, **kwargs)

        start_time = datetime.utcnow()

        try:
            claude_provider = self.providers["claude"]
            result = await claude_provider.generate_response_with_thinking(
                messages,
                thinking_budget=thinking_budget,
                **kwargs
            )

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            return {
                "response": result["response"],
                "thinking": result.get("thinking"),
                "used_thinking": result.get("used_thinking", False),
                "provider": "claude",
                "timestamp": end_time.isoformat(),
                "duration_seconds": duration,
                "success": True
            }

        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            logger.error(
                "Extended thinking generation failed",
                provider="claude",
                error=str(e),
                duration=duration
            )

            # Fall back to regular generation
            logger.info("Falling back to regular response generation after thinking failure")
            return await self.generate_response(messages, provider="claude", **kwargs)

    def _should_use_extended_thinking(
        self,
        confidence: float,
        has_clarification: bool,
        conversation_length: int,
        user_message: str
    ) -> bool:
        """
        Determine if extended thinking should be used for this request.

        Conditions for extended thinking:
        1. Low confidence (< 0.7) - ambiguous intent
        2. Pending clarifications - user didn't provide enough info
        3. Long conversations (> 20 messages) - complex context

        Args:
            confidence: Intent detection confidence (0.0 to 1.0)
            has_clarification: Whether clarification is needed
            conversation_length: Number of messages in conversation
            user_message: The user's current message

        Returns:
            True if extended thinking should be used
        """
        # Condition 1: Low confidence in intent detection
        if confidence < 0.7:
            logger.info(
                "Using extended thinking: low confidence",
                confidence=confidence
            )
            return True

        # Condition 2: Clarification needed
        if has_clarification:
            logger.info("Using extended thinking: clarification pending")
            return True

        # Condition 3: Long conversation (complex context)
        if conversation_length > 20:
            logger.info(
                "Using extended thinking: long conversation",
                message_count=conversation_length
            )
            return True

        # Condition 4: Complex references in user message
        complex_reference_patterns = [
            "the one",
            "that one",
            "first one",
            "second one",
            "third one",
            "last one",
            "the email about",
            "the contact from",
            "the project with",
        ]
        message_lower = user_message.lower()
        for pattern in complex_reference_patterns:
            if pattern in message_lower:
                logger.info(
                    "Using extended thinking: complex reference detected",
                    pattern=pattern
                )
                return True

        return False

    async def generate_crm_response(
        self,
        user_message: str,
        user_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]] = None,
        provider: Optional[str] = None,
        confidence: float = 1.0,
        has_clarification: bool = False
    ) -> Dict[str, Any]:
        """
        Generate CRM-specific response with intelligent context management.

        Features:
        - Tiered context management (recent/older/ancient messages)
        - Token-aware context window sizing
        - Data payload preservation
        - Follow-up question support
        - Extended thinking for complex situations (Claude only)

        Args:
            user_message: The user's message/command
            user_context: Context about the user (name, recent data, etc.)
            conversation_history: Previous messages in conversation
            provider: AI provider to use
            confidence: Intent detection confidence (0.0 to 1.0)
            has_clarification: Whether clarification is pending

        Returns:
            AI response with CRM context
        """
        from .conversation_memory import conversation_memory

        # Build concise system prompt for CRM
        system_prompt = f"""You are a CRM assistant for {user_context.get('name', 'User')}.

YOUR CAPABILITIES - YOU CAN:
✓ Create, update, and delete contacts, organizations, projects, and tasks
✓ Search and list all CRM data
✓ Access Microsoft 365 emails, calendar, and files (if connected)
✓ Execute database operations and report results

HOW IT WORKS:
- Users ask you to perform operations
- The system executes them automatically BEFORE generating your response
- You receive the execution results in this message
- You report the results to the user (success or failure)
- You DO have these capabilities - never deny them

CRITICAL: When you see "✓ CONFIRMED:" in a message, the operation ALREADY succeeded.
Report success confidently. You're not "unable" to do anything - you're reporting what was done.

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

Connected Integrations:
{self._get_integration_capabilities_text(user_context)}

Email Data Structure (Microsoft 365) - CRITICAL DATA BOUNDARIES:

Two types of email data you will see:

1. EMAIL SEARCH RESULTS (lists):
   - Contains: Subject, From, Date, Body Preview (~200 chars ONLY)
   - Label: "Body Preview (TRUNCATED)"
   - YOU DO NOT HAVE FULL BODY CONTENT for email lists
   - If asked for full content, say you only have preview and offer to retrieve specific email
   - NEVER claim to know full email content from search results
   - NEVER extrapolate or make up content beyond the preview

2. SINGLE EMAIL RETRIEVAL (individual):
   - Contains: Subject, From, Date, FULL BODY CONTENT (complete text)
   - Label: "FULL BODY CONTENT (COMPLETE)"
   - You HAVE complete email content for single email retrievals
   - Can discuss and summarize full content confidently
   - Content is explicitly shown in "FULL BODY CONTENT:" section

CRITICAL ANTI-HALLUCINATION RULE - EMAIL BODY RETRIEVAL:
When user asks to "retrieve", "show full body", "get the content", or says "yes" after you offered to retrieve:
❌ FORBIDDEN: Making up or fabricating email body content
❌ FORBIDDEN: Expanding on the preview to guess full content
❌ FORBIDDEN: Responding with ANY text that claims to be the email body
✅ REQUIRED: Wait for the system to execute outlook_get_email tool
✅ REQUIRED: Check the conversation for "==== EMAIL DATA START ====" marker
✅ REQUIRED: Only discuss email body if you see the FULL BODY CONTENT section

If user asks "show me the full email" and you don't see tool execution results:
SAY: "I need to retrieve the full email for you. One moment..."
THEN: The system will execute the tool and provide the full content
WAIT: For the next message with full email data before discussing content

DATA ACCURACY RULES:
- Only report data that's explicitly provided in this conversation
- If an operation failed (you see "ERROR:"), report the failure honestly
- If an operation succeeded (you see "✓ CONFIRMED:"), report the success confidently
- Don't invent or estimate numbers - use exact counts provided
- Don't make up email content, contact details, or any data not shown to you

IMPORTANT: Reporting successful operations is NOT hallucinating.
If you see "✓ CONFIRMED: Contact created", it really was created - say so confidently.

CRITICAL: HOW CRUD OPERATIONS WORK
When users ask you to CREATE, UPDATE, or DELETE anything:

1. The system automatically executes the operation BEFORE you see this message
2. Look for execution results in this message - they are ALREADY included
3. If you see "✓ CONFIRMED: [operation] succeeded", the operation DID succeed
4. If you see "ERROR:" or "Failed to", the operation DID fail

How to respond:
✅ If you see "✓ CONFIRMED: Contact 'Luke' deleted successfully"
   → Confidently say: "I've successfully deleted Luke's contact."

✅ If you see "ERROR: Contact not found"
   → Say: "I couldn't find a contact named Luke. Could you check the name?"

✅ If you see "✓ CONFIRMED: Contact 'John Doe' created successfully with ID 123"
   → Confidently say: "I've created the contact for John Doe."

❌ NEVER say "I don't have the ability to do that" - you DO have the ability
❌ NEVER say "I need to process that" - it's ALREADY processed
❌ NEVER make up results if you see an error - report the error honestly
❌ NEVER claim success if you see an error message

The execution happens BEFORE you respond. You're reporting results, not initiating actions.

Context Handling:
- When users ask follow-up questions (e.g., "tell me more about that", "what's the content", "show me the body"),
  they are referring to data from previous messages in this conversation
- ALWAYS check conversation history before claiming you need to retrieve data
- If data was already shown in previous messages, reference it directly and confidently
- Never re-request or claim you cannot access data that's already in the conversation
- Be natural when referencing previous data - act like you remember what we just discussed

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

        # Build message history with smart context pruning
        from .conversation_memory import SmartContextPruner

        messages = [{"role": "system", "content": system_prompt}]

        # Use 8000 tokens for conversation history, leaving room for system prompt and response
        context_budget = 8000

        # Get adaptive context size based on conversation length
        if conversation_history:
            # Use smart pruning for any conversation
            pruned_history, tokens_used, was_pruned = SmartContextPruner.prune_messages(
                conversation_history,
                max_tokens=context_budget,
                preserve_recent=min(15, len(conversation_history))  # Keep up to 15 recent
            )

            if was_pruned:
                logger.info(
                    "Applied smart context pruning",
                    original_messages=len(conversation_history),
                    pruned_messages=len(pruned_history),
                    tokens_used=tokens_used,
                    budget=context_budget
                )

            messages.extend(pruned_history)
        else:
            conversation_history = []

        # Check if we should inject relevant context for follow-up questions
        relevant_context = conversation_memory.get_relevant_context(user_message)
        if relevant_context:
            user_message = relevant_context + "\n\n" + user_message
            logger.debug("Injected relevant context for follow-up question")

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Estimate total tokens (rough)
        total_tokens = sum(conversation_memory.estimate_tokens(msg.get("content", "")) for msg in messages)

        # Determine conversation length for extended thinking check
        conversation_length = len(conversation_history) if conversation_history else 0

        # Check if extended thinking should be used
        use_thinking = self._should_use_extended_thinking(
            confidence=confidence,
            has_clarification=has_clarification,
            conversation_length=conversation_length,
            user_message=user_message
        )

        logger.info(
            "Generating CRM response",
            message_count=len(messages),
            estimated_tokens=total_tokens,
            provider=provider or self.default_provider,
            use_extended_thinking=use_thinking,
            confidence=confidence
        )

        # Generate response - use extended thinking for complex situations
        if use_thinking and (provider or self.default_provider) == "claude":
            # Adjust thinking budget based on complexity
            thinking_budget = 10000
            if conversation_length > 30:
                thinking_budget = 15000  # More thinking for very long conversations
            elif confidence < 0.5:
                thinking_budget = 12000  # More thinking for very low confidence

            return await self.generate_response_with_thinking(
                messages,
                thinking_budget=thinking_budget,
                provider="claude"
            )
        else:
            return await self.generate_response(messages, provider=provider)

    def _get_adaptive_context_size(self, conversation_length: int) -> int:
        """
        Determine adaptive context window size based on conversation length.

        Args:
            conversation_length: Number of messages in conversation

        Returns:
            Number of messages to include in context
        """
        if conversation_length < 20:
            return conversation_length  # Show all for short conversations
        elif conversation_length < 50:
            return 25  # Show last 25 for medium conversations
        else:
            return 30  # Show last 30 + summaries for long conversations

    def _get_integration_capabilities_text(self, user_context: Dict[str, Any]) -> str:
        """Generate text describing user's connected integration capabilities."""

        capabilities = []

        # Check explicit integration status first (passed from chat.py)
        integrations = user_context.get('integrations', {})
        ms365_status = integrations.get('microsoft365', {})

        if ms365_status.get('connected'):
            ms365_capabilities = []
            if ms365_status.get('sync_emails'):
                ms365_capabilities.extend([
                    "retrieve and search your Outlook emails (including body content)",
                    "get your latest/most recent email",
                    "find new/unread emails",
                    "read full email content with attachments",
                    "send emails",
                    "create drafts"
                ])
            if ms365_status.get('sync_calendars'):
                ms365_capabilities.extend([
                    "view calendar events", "create meetings", "schedule events"
                ])
            if ms365_status.get('sync_files'):
                ms365_capabilities.extend([
                    "search SharePoint documents", "access OneDrive files", "list Teams files"
                ])

            if ms365_capabilities:
                capabilities.append(f"- Microsoft 365 Connected: I can {', '.join(ms365_capabilities)}")
                capabilities.append("  Note: I have full access to your Microsoft 365 data through secure integration.")
            else:
                # MS365 is connected but no sync options enabled
                capabilities.append("- Microsoft 365 Connected: Basic access enabled. Enable email/calendar/files sync in Settings for full capabilities.")

        # Fallback: check user object directly (for backward compatibility)
        if not capabilities:
            user = user_context.get('user')
            if user and hasattr(user, 'integrations') and user.integrations:
                for integration in user.integrations:
                    if integration.service_type == "microsoft365" and integration.is_connected:
                        ms365_capabilities = []
                        if integration.sync_emails:
                            ms365_capabilities.extend([
                                "retrieve and search your Outlook emails",
                                "get your latest/most recent email",
                                "send emails"
                            ])
                        if integration.sync_calendars:
                            ms365_capabilities.extend([
                                "view calendar events", "create meetings"
                            ])
                        if integration.sync_files:
                            ms365_capabilities.extend([
                                "search SharePoint documents", "access OneDrive files"
                            ])

                        if ms365_capabilities:
                            capabilities.append(f"- Microsoft 365 Connected: I can {', '.join(ms365_capabilities)}")
                        break

        if not capabilities:
            return "- No external integrations connected. Connect Microsoft 365 in Settings to access email, calendar, and files."

        return "\n".join(capabilities)


# Global AI service instance
ai_service = AIService()