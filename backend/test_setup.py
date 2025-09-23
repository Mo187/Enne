#!/usr/bin/env python3
"""
Simple test script to verify the CRM application setup
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import engine, create_tables
from app.core.config import settings
from sqlalchemy import text


async def test_database_connection():
    """Test database connectivity"""
    print("🔍 Testing database connection...")
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ PostgreSQL connected: {version}")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


async def test_table_creation():
    """Test table creation"""
    print("🔍 Testing table creation...")
    try:
        await create_tables()
        print("✅ Tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Table creation failed: {e}")
        return False


async def test_redis_connection():
    """Test Redis connectivity"""
    print("🔍 Testing Redis connection...")
    try:
        import redis
        r = redis.from_url(settings.redis_url)
        r.ping()
        print("✅ Redis connected successfully")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False


async def test_api_imports():
    """Test API module imports"""
    print("🔍 Testing API imports...")
    try:
        from app.main import app
        from app.api.v1.auth import router
        from app.models import User, Contact, Organization, Project, Task, Integration
        print("✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting CRM Application Setup Tests\n")

    tests = [
        ("API Imports", test_api_imports),
        ("Database Connection", test_database_connection),
        ("Redis Connection", test_redis_connection),
        ("Table Creation", test_table_creation),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! Your CRM setup is working correctly.")
        print("\n🚀 Next steps:")
        print("   1. Start the FastAPI server: uvicorn app.main:app --reload")
        print("   2. Visit http://localhost:8000 to see the dashboard")
        print("   3. Visit http://localhost:8000/docs for API documentation")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)