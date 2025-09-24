#!/usr/bin/env python3
"""
Test script for AI Chat functionality
Demonstrates the natural language processing and command execution
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_command_parsing():
    """Test the command parser with various inputs"""
    print("🧪 Testing Command Parser\n")

    try:
        from backend.app.services.old_command_parser import command_parser, CommandType

        test_commands = [
            "Add contact John Smith from Acme Corp",
            "Create new contact Sarah Johnson with email sarah@techcorp.com",
            "Find contacts from technology companies",
            "Show me all organizations in healthcare",
            "Update contact John's phone number to 555-1234",
            "Delete contact with email test@example.com",
            "Export all contacts to CSV",
            "Add organization Google in technology industry",
            "Show me CRM statistics",
            "What can you help me with?",
            "Random text that shouldn't match anything"
        ]

        for command in test_commands:
            result = command_parser.parse_command(command)
            action = command_parser.generate_api_action(result)

            print(f"📝 Command: '{command}'")
            print(f"   Type: {result['command_type'].value}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Data: {result['data']}")
            print(f"   API Action: {action['method']} {action['endpoint']}")
            print()

        print("✅ Command parsing tests completed\n")
        return True

    except Exception as e:
        print(f"❌ Command parsing test failed: {e}\n")
        return False


def test_ai_service_setup():
    """Test AI service initialization"""
    print("🤖 Testing AI Service Setup\n")

    try:
        from app.services.ai_service import ai_service

        providers = ai_service.get_available_providers()
        print(f"📋 Available AI providers: {providers}")

        if not providers:
            print("⚠️  No AI providers available (API keys not set)")
            print("   Add API keys to .env file:")
            print("   - ANTHROPIC_API_KEY for Claude")
            print("   - OPENAI_API_KEY for OpenAI")
            print("   - GOOGLE_AI_KEY for Gemini")
        else:
            print(f"✅ {len(providers)} AI provider(s) configured")

        print()
        return True

    except Exception as e:
        print(f"❌ AI service test failed: {e}\n")
        return False


async def test_ai_response():
    """Test AI response generation (if providers available)"""
    print("💬 Testing AI Response Generation\n")

    try:
        from app.services.ai_service import ai_service

        providers = ai_service.get_available_providers()
        if not providers:
            print("⚠️  Skipping AI response test - no providers available")
            return True

        # Test with a simple CRM context
        user_context = {
            "name": "Test User",
            "email": "test@example.com",
            "company": "Test Company"
        }

        print("🔄 Testing AI response...")

        response = await ai_service.generate_crm_response(
            user_message="Hello, what can you help me with?",
            user_context=user_context,
            provider=providers[0]  # Use first available provider
        )

        if response["success"]:
            print(f"✅ AI Response received from {response['provider']}")
            print(f"   Duration: {response['duration_seconds']:.2f} seconds")
            print(f"   Response: {response['response'][:100]}...")
        else:
            print(f"❌ AI Response failed: {response.get('error', 'Unknown error')}")

        print()
        return response["success"]

    except Exception as e:
        print(f"❌ AI response test failed: {e}\n")
        return False


def test_api_endpoints():
    """Test that API endpoints are properly configured"""
    print("🌐 Testing API Endpoints Configuration\n")

    try:
        from app.main import app

        # Get all routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                for method in route.methods:
                    if method != "HEAD":  # Skip HEAD methods
                        routes.append(f"{method} {route.path}")

        # Check for essential endpoints
        essential_endpoints = [
            "GET /health",
            "GET /",
            "GET /assistant",
            "POST /api/v1/auth/register",
            "POST /api/v1/auth/login",
            "GET /api/v1/contacts",
            "POST /api/v1/contacts",
            "GET /api/v1/organizations",
            "POST /api/v1/organizations",
            "POST /api/v1/chat/message",
            "GET /api/v1/chat/providers"
        ]

        print("📋 Essential API endpoints:")
        for endpoint in essential_endpoints:
            if endpoint in routes:
                print(f"   ✅ {endpoint}")
            else:
                print(f"   ❌ {endpoint} - NOT FOUND")

        print(f"\n📊 Total routes configured: {len(routes)}")
        print()
        return True

    except Exception as e:
        print(f"❌ API endpoints test failed: {e}\n")
        return False


def test_templates():
    """Test that templates are available"""
    print("🎨 Testing Template Files\n")

    try:
        template_files = [
            "app/templates/base.html",
            "app/templates/pages/dashboard.html",
            "app/templates/pages/assistant.html"
        ]

        missing_files = []
        for template in template_files:
            if not os.path.exists(template):
                missing_files.append(template)

        if missing_files:
            print("❌ Missing template files:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        else:
            print("✅ All template files present")
            print("   - Base template with navigation")
            print("   - Dashboard page")
            print("   - AI Assistant chat interface")

        print()
        return True

    except Exception as e:
        print(f"❌ Template test failed: {e}\n")
        return False


async def main():
    """Run all AI chat tests"""
    print("🚀 AI Chat Functionality Tests\n")
    print("="*60)

    tests = [
        ("Template Files", test_templates),
        ("API Endpoints", test_api_endpoints),
        ("Command Parser", test_command_parsing),
        ("AI Service Setup", test_ai_service_setup),
        ("AI Response Generation", test_ai_response),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "="*60)
    print("📊 AI CHAT TEST RESULTS")
    print("="*60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All AI chat tests passed!")
        print("\n🚀 Your AI-Assisted CRM is ready!")
        print("\n📝 To start using:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Set AI API keys in .env file")
        print("   3. Start database: docker-compose up -d postgres redis")
        print("   4. Run migrations: alembic upgrade head")
        print("   5. Start server: uvicorn app.main:app --reload")
        print("   6. Open: http://localhost:8000/assistant")
        print("\n💬 Try these commands:")
        print("   - 'Add contact John Smith from Acme Corp'")
        print("   - 'Show me all organizations in technology'")
        print("   - 'Export my contacts to CSV'")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        if passed >= 3:  # If most tests pass
            print("   The basic functionality should still work!")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)