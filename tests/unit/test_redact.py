#!/usr/bin/env python3
"""
Unit tests for redaction utilities
Tests pattern coverage for email, API key, and secret redaction
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.redact import (
    redact_email, redact_api_keys, redact_secrets, redact,
    redact_dict, is_sensitive_field, safe_log_format
)


class TestEmailRedaction:
    """Test email address redaction"""
    
    def test_redact_single_email(self):
        """Test redacting a single email address"""
        text = "Contact us at support@example.com for help"
        result = redact_email(text)
        assert "support@example.com" not in result
        assert "[REDACTED]" in result
        assert result == "Contact us at [REDACTED] for help"
    
    def test_redact_multiple_emails(self):
        """Test redacting multiple email addresses"""
        text = "Send to alice@company.com and bob@test.org"
        result = redact_email(text)
        assert "alice@company.com" not in result
        assert "bob@test.org" not in result
        assert result.count("[REDACTED]") == 2
    
    def test_redact_email_in_context(self):
        """Test email redaction within larger context"""
        text = "User john.doe+test@gmail.com logged in from 192.168.1.1"
        result = redact_email(text)
        assert "john.doe+test@gmail.com" not in result
        assert "[REDACTED]" in result
        assert "192.168.1.1" in result  # IP should remain
    
    def test_no_email_to_redact(self):
        """Test text without emails remains unchanged"""
        text = "This is just normal text without any emails"
        result = redact_email(text)
        assert result == text
    
    def test_malformed_email_patterns(self):
        """Test handling of malformed email-like patterns"""
        text = "Not email: user@, @domain.com, user@domain"
        result = redact_email(text)
        # These shouldn't be redacted as they're not valid emails
        assert result == text


class TestAPIKeyRedaction:
    """Test API key redaction"""
    
    def test_redact_long_alphanumeric(self):
        """Test redacting long alphanumeric strings (potential API keys)"""
        text = "API key: sk_test_1234567890abcdef1234567890"
        result = redact_api_keys(text)
        assert "sk_test_1234567890abcdef1234567890" not in result
        assert "sk_t***7890" in result  # Partial mask
    
    def test_preserve_short_strings(self):
        """Test that short strings are not redacted"""
        text = "Version 1.2.3 and build abc123def"
        result = redact_api_keys(text)
        assert result == text  # Should be unchanged
    
    def test_preserve_numbers(self):
        """Test that pure numbers are not redacted"""
        text = "Transaction ID: 12345678901234567890123456789"
        result = redact_api_keys(text)
        assert result == text  # Numbers should be preserved
    
    def test_github_token_pattern(self):
        """Test GitHub token-like patterns"""
        text = "Token: ghp_1234567890abcdef1234567890abcdef12345678"
        result = redact_api_keys(text)
        assert "ghp_1234567890abcdef1234567890abcdef12345678" not in result
        assert "ghp_***5678" in result


class TestSecretRedaction:
    """Test secret pattern redaction"""
    
    def test_redact_password_assignment(self):
        """Test redacting password assignments"""
        text = "password=secret123"
        result = redact_secrets(text)
        assert "secret123" not in result
        assert "password=[REDACTED]" in result
    
    def test_redact_api_key_assignment(self):
        """Test redacting API key assignments"""
        text = "API_KEY: abc123def456"
        result = redact_secrets(text)
        assert "abc123def456" not in result
        assert "[REDACTED]" in result
    
    def test_redact_bearer_token(self):
        """Test redacting Bearer tokens"""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = redact_secrets(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "[REDACTED]" in result
    
    def test_redact_basic_auth(self):
        """Test redacting Basic auth"""
        text = "Authorization: Basic dXNlcjpwYXNzd29yZA=="
        result = redact_secrets(text)
        assert "dXNlcjpwYXNzd29yZA==" not in result
        assert "[REDACTED]" in result
    
    def test_redact_openai_key(self):
        """Test redacting OpenAI-style keys"""
        text = "OPENAI_API_KEY=sk-1234567890abcdef1234567890abcdef1234567890abcdef"
        result = redact_secrets(text)
        assert "sk-1234567890abcdef1234567890abcdef1234567890abcdef" not in result
        assert "[REDACTED]" in result


class TestCombinedRedaction:
    """Test the main redact() function with multiple patterns"""
    
    def test_redact_mixed_content(self):
        """Test redacting content with multiple sensitive patterns"""
        text = """
        User: admin@company.com
        Password: mySecret123
        API Key: sk_test_1234567890abcdef1234567890
        Normal text here
        """
        result = redact(text)
        
        assert "admin@company.com" not in result
        assert "mySecret123" not in result
        assert "sk_test_1234567890abcdef1234567890" not in result
        assert "Normal text here" in result
        assert "[REDACTED]" in result
    
    def test_redact_non_string_input(self):
        """Test redacting non-string input"""
        result = redact(12345)
        assert result == "12345"
        
        result = redact(None)
        assert result == "None"
        
        result = redact(['test@email.com', 'password=secret'])
        assert "[REDACTED]" in result


class TestDictRedaction:
    """Test dictionary redaction"""
    
    def test_redact_dict_values(self):
        """Test redacting sensitive values in dictionaries"""
        data = {
            'username': 'john',
            'email': 'john@example.com',
            'api_key': 'sk_test_1234567890abcdef',
            'normal_field': 'normal_value'
        }
        
        result = redact_dict(data)
        
        assert result['username'] == 'john'
        assert result['email'] == '[REDACTED]'
        assert result['normal_field'] == 'normal_value'
        assert '[REDACTED]' in result['api_key'] or result['api_key'] == '[REDACTED]'
    
    def test_redact_nested_dict(self):
        """Test redacting nested dictionaries"""
        data = {
            'user': {
                'email': 'user@test.com',
                'credentials': {
                    'password': 'secret123'
                }
            },
            'config': {
                'timeout': 30
            }
        }
        
        result = redact_dict(data)
        
        assert result['user']['email'] == '[REDACTED]'
        assert '[REDACTED]' in result['user']['credentials']['password']
        assert result['config']['timeout'] == 30
    
    def test_redact_dict_with_lists(self):
        """Test redacting dictionaries containing lists"""
        data = {
            'emails': ['user1@test.com', 'user2@test.com'],
            'numbers': [1, 2, 3],
            'mixed': ['normal', 'admin@company.com', 42]
        }
        
        result = redact_dict(data)
        
        assert all('[REDACTED]' == email for email in result['emails'])
        assert result['numbers'] == [1, 2, 3]
        assert '[REDACTED]' in result['mixed']
        assert 'normal' in result['mixed']
        assert 42 in result['mixed']


class TestSensitiveFieldDetection:
    """Test sensitive field name detection"""
    
    def test_common_sensitive_fields(self):
        """Test detection of common sensitive field names"""
        sensitive_fields = [
            'password', 'passwd', 'pwd', 'secret', 'token', 'key',
            'api_key', 'access_token', 'refresh_token', 'private_key'
        ]
        
        for field in sensitive_fields:
            assert is_sensitive_field(field)
            assert is_sensitive_field(field.upper())
            assert is_sensitive_field(f'user_{field}')
    
    def test_non_sensitive_fields(self):
        """Test that normal fields are not flagged as sensitive"""
        normal_fields = [
            'username', 'email', 'name', 'id', 'timestamp',
            'value', 'count', 'status', 'type'
        ]
        
        for field in normal_fields:
            assert not is_sensitive_field(field)


class TestSafeLogFormat:
    """Test safe log formatting"""
    
    def test_safe_log_basic(self):
        """Test basic safe log formatting"""
        result = safe_log_format("User login", user="john", email="john@test.com")
        
        assert "User login" in result
        assert "user=john" in result
        assert "john@test.com" not in result
        assert "[REDACTED]" in result
    
    def test_safe_log_with_sensitive_fields(self):
        """Test safe log formatting with sensitive field names"""
        result = safe_log_format("API call", api_key="secret123", timeout=30)
        
        assert "api_key=[REDACTED]" in result
        assert "timeout=30" in result
        assert "secret123" not in result
    
    def test_safe_log_message_redaction(self):
        """Test that the base message is also redacted"""
        result = safe_log_format("Failed to authenticate user@test.com")
        
        assert "user@test.com" not in result
        assert "[REDACTED]" in result


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_string(self):
        """Test redaction of empty string"""
        assert redact("") == ""
        assert redact_email("") == ""
        assert redact_api_keys("") == ""
        assert redact_secrets("") == ""
    
    def test_none_input(self):
        """Test redaction of None input"""
        assert redact(None) == "None"
    
    def test_very_long_string(self):
        """Test redaction of very long strings"""
        long_text = "normal text " * 1000 + "secret@email.com"
        result = redact(long_text)
        assert "secret@email.com" not in result
        assert "[REDACTED]" in result
        assert result.count("normal text") == 1000
    
    def test_unicode_content(self):
        """Test redaction with unicode content"""
        text = "Email: tëst@éxample.com and key: ñörmàl123tèxt"
        result = redact(text)
        assert "tëst@éxample.com" not in result
        assert "[REDACTED]" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])