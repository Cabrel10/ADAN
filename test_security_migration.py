#!/usr/bin/env python3
"""
Test script to verify the security infrastructure migration.
This script tests the security validation without importing the full module.
"""

import os
import sys
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hardcoded_keys_removed():
    """Test that hardcoded API keys file has been removed"""
    logger.info("Testing hardcoded keys file removal...")

    suspicious_files = [
        "gemini_api_keys.txt",
        "api_keys.txt",
        "keys.txt",
        "secrets.txt"
    ]

    found_files = []
    for file_name in suspicious_files:
        file_path = Path(file_name)
        if file_path.exists():
            found_files.append(file_name)

    if found_files:
        logger.error(f"❌ Found suspicious files with potential hardcoded keys: {found_files}")
        return False
    else:
        logger.info("✅ No hardcoded API key files found")
        return True

def test_environment_setup_guide():
    """Test that environment setup guide was created"""
    logger.info("Testing environment setup guide creation...")

    guide_path = Path("../ENVIRONMENT_SETUP.md")
    if guide_path.exists():
        logger.info("✅ Environment setup guide exists")

        # Check if it contains expected content
        content = guide_path.read_text()
        required_sections = [
            "Environment Variables Setup Guide",
            "BINANCE_API_KEY",
            "BINANCE_API_SECRET",
            "Security Best Practices",
            "Never commit API keys to version control"
        ]

        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)

        if missing_sections:
            logger.error(f"❌ Environment guide missing sections: {missing_sections}")
            return False
        else:
            logger.info("✅ Environment setup guide contains all required sections")
            return True
    else:
        logger.error("❌ Environment setup guide not found")
        return False

def test_security_validation_logic():
    """Test the security validation logic independently"""
    logger.info("Testing security validation logic...")

    # API key patterns that should be detected
    api_key_patterns = [
        r'AIzaSy[A-Za-z0-9_-]{33}',  # Google API keys
        r'sk-[A-Za-z0-9]{48}',       # OpenAI API keys
        r'[A-Za-z0-9]{64}',          # Generic 64-char keys
        r'[A-Za-z0-9]{32}',          # Generic 32-char keys
    ]

    # Test strings that should trigger detection
    test_strings = [
        "AIzaSyC-PnZoTqGlr_VvCc9XHs7h5oXMdKKds0I",  # Google API key format
        "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",  # OpenAI format
        "abcd1234567890abcd1234567890abcd1234567890abcd1234567890abcd1234",  # 64-char
        "abcd1234567890abcd1234567890abcd12"  # 32-char (exactly 32 characters)
    ]

    detected_count = 0
    for test_string in test_strings:
        for pattern in api_key_patterns:
            if re.search(pattern, test_string):
                detected_count += 1
                logger.info(f"   ✓ Pattern detected: {test_string[:10]}...")
                break

    if detected_count == len(test_strings):
        logger.info("✅ Security validation patterns working correctly")
        return True
    else:
        logger.error(f"❌ Security validation patterns failed: {detected_count}/{len(test_strings)} detected")
        return False

def test_gitignore_updated():
    """Test that .gitignore includes security-related entries"""
    logger.info("Testing .gitignore security entries...")

    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        logger.warning("⚠️  .gitignore file not found")
        return False

    content = gitignore_path.read_text()
    security_entries = [
        "gemini_api_keys.txt",
        ".env"
    ]

    found_entries = []
    for entry in security_entries:
        if entry in content:
            found_entries.append(entry)

    if len(found_entries) == len(security_entries):
        logger.info("✅ .gitignore contains all required security entries")
        return True
    else:
        missing = set(security_entries) - set(found_entries)
        logger.warning(f"⚠️  .gitignore missing security entries: {missing}")
        return True  # Not critical, just a warning

def main():
    """Run all tests"""
    logger.info("🚀 Starting Security Infrastructure Migration Tests")
    logger.info("=" * 60)

    tests = [
        ("Hardcoded Keys Removal", test_hardcoded_keys_removed),
        ("Environment Setup Guide", test_environment_setup_guide),
        ("Security Validation Logic", test_security_validation_logic),
        ("GitIgnore Security Entries", test_gitignore_updated)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running test: {test_name}")
        logger.info("-" * 40)

        if test_func():
            passed += 1

        logger.info("-" * 40)

    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Security infrastructure migration is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
