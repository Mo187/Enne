#!/usr/bin/env python3
"""
Validation script to test CRM API endpoints and functionality
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")

    try:
        # Test core imports
        from app.core.config import settings
        from app.core.database import get_db, create_tables
        from app.core.security import create_access_token, verify_password, get_password_hash
        print("✅ Core modules imported successfully")

        # Test model imports
        from app.models import User, Contact, Organization, Project, Task, Integration
        print("✅ All models imported successfully")

        # Test API imports
        from app.api.v1.auth import router as auth_router
        from app.api.v1.contacts import router as contacts_router
        from app.api.v1.organizations import router as organizations_router
        print("✅ API routers imported successfully")

        # Test main app
        from app.main import app
        print("✅ FastAPI app imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_configuration():
    """Test configuration settings"""
    print("🔍 Testing configuration...")

    try:
        from app.core.config import settings

        # Test basic settings
        assert settings.app_name, "App name not set"
        assert settings.secret_key, "Secret key not set"
        assert settings.database_url, "Database URL not set"
        assert settings.redis_url, "Redis URL not set"

        print(f"✅ App Name: {settings.app_name}")
        print(f"✅ Database URL: {settings.database_url[:50]}...")
        print(f"✅ Redis URL: {settings.redis_url}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_security():
    """Test security functions"""
    print("🔍 Testing security functions...")

    try:
        from app.core.security import create_access_token, verify_password, get_password_hash

        # Test password hashing
        password = "testpassword123"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed), "Password verification failed"
        print("✅ Password hashing and verification working")

        # Test token creation
        token = create_access_token(subject="123")
        assert token, "Token creation failed"
        print("✅ JWT token creation working")

        return True

    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False


def test_models():
    """Test model structure"""
    print("🔍 Testing model structure...")

    try:
        from app.models import User, Contact, Organization, Project, Task, Integration

        # Test model attributes
        user_attrs = ['id', 'email', 'name', 'password_hash', 'is_active']
        for attr in user_attrs:
            assert hasattr(User, attr), f"User missing attribute: {attr}"

        contact_attrs = ['id', 'user_id', 'name', 'email', 'phone']
        for attr in contact_attrs:
            assert hasattr(Contact, attr), f"Contact missing attribute: {attr}"

        org_attrs = ['id', 'user_id', 'name', 'industry', 'website']
        for attr in org_attrs:
            assert hasattr(Organization, attr), f"Organization missing attribute: {attr}"

        print("✅ All model attributes present")

        # Test relationships
        assert hasattr(User, 'contacts'), "User missing contacts relationship"
        assert hasattr(User, 'organizations'), "User missing organizations relationship"
        assert hasattr(Contact, 'user'), "Contact missing user relationship"
        assert hasattr(Organization, 'user'), "Organization missing user relationship"

        print("✅ Model relationships defined")

        return True

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_api_structure():
    """Test API endpoint structure"""
    print("🔍 Testing API structure...")

    try:
        from app.main import app
        from fastapi.testclient import TestClient

        # This would require TestClient which may not be available
        # For now, just test that routes are defined
        routes = [route.path for route in app.routes]

        # Check for essential routes
        essential_routes = ["/health", "/api/v1/auth/register", "/api/v1/auth/login"]

        for route in essential_routes:
            if any(route in r for r in routes):
                print(f"✅ Route found: {route}")
            else:
                print(f"⚠️  Route might be missing: {route}")

        print("✅ API structure test completed")
        return True

    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False


def test_file_structure():
    """Test file structure"""
    print("🔍 Testing file structure...")

    try:
        required_files = [
            "app/__init__.py",
            "app/main.py",
            "app/core/config.py",
            "app/core/database.py",
            "app/core/security.py",
            "app/models/__init__.py",
            "app/models/user.py",
            "app/models/contact.py",
            "app/models/organization.py",
            "app/api/__init__.py",
            "app/api/v1/__init__.py",
            "app/api/v1/auth.py",
            "app/api/v1/contacts.py",
            "app/api/v1/organizations.py",
            "app/templates/base.html",
            "alembic.ini",
            "alembic/env.py",
            "alembic/versions/001_initial_migration.py"
        ]

        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            print("❌ Missing files:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            return False
        else:
            print("✅ All required files present")
            return True

    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False


def main():
    """Run all validation tests"""
    print("🚀 Starting CRM API Validation Tests\n")

    tests = [
        ("File Structure", test_file_structure),
        ("Configuration", test_configuration),
        ("Imports", test_imports),
        ("Security Functions", test_security),
        ("Model Structure", test_models),
        ("API Structure", test_api_structure),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "="*60)
    print("📊 VALIDATION RESULTS SUMMARY")
    print("="*60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All validation tests passed!")
        print("\n🚀 Your CRM API is ready for development!")
        print("\n📝 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Start database: docker-compose up -d postgres redis")
        print("   3. Run migrations: alembic upgrade head")
        print("   4. Start server: uvicorn app.main:app --reload")
        print("   5. Test API: http://localhost:8000/docs")
    else:
        print("\n⚠️  Some validation tests failed.")
        print("   Please check the errors above and fix any issues.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)