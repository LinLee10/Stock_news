#!/usr/bin/env python3
"""
Security redaction utilities for log sanitization
Scrubs sensitive data from logs including emails, API keys, and secrets
"""

import re
from typing import Any, Union


# Patterns for sensitive data detection
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
API_KEY_PATTERN = re.compile(r'[A-Za-z0-9]{20,}')  # 20+ alphanumeric strings (likely API keys)
SECRET_PATTERNS = [
    re.compile(r'(password|passwd|pwd|secret|token|key|auth)[\'"\s]*[:=][\'"\s]*([^\s\'"&<>]+)', re.IGNORECASE),
    re.compile(r'Bearer\s+[A-Za-z0-9\-_]{10,}', re.IGNORECASE),
    re.compile(r'Basic\s+[A-Za-z0-9+/=]{10,}', re.IGNORECASE),
    re.compile(r'sk-[A-Za-z0-9]{40,}'),  # OpenAI-style keys
    re.compile(r'ghp_[A-Za-z0-9]{36}'),  # GitHub personal access tokens
    re.compile(r'gho_[A-Za-z0-9]{36}'),  # GitHub OAuth tokens
]

# Common redaction mask
REDACTED_MASK = "[REDACTED]"
PARTIAL_MASK = "***"


def redact_email(text: str) -> str:
    """
    Redact email addresses from text
    
    Args:
        text: Input text potentially containing emails
        
    Returns:
        Text with emails replaced by [REDACTED]
    """
    return EMAIL_PATTERN.sub(REDACTED_MASK, text)


def redact_api_keys(text: str) -> str:
    """
    Redact potential API keys (20+ character alphanumeric strings)
    
    Args:
        text: Input text potentially containing API keys
        
    Returns:
        Text with potential API keys replaced by [REDACTED]
    """
    def replace_key(match):
        key = match.group(0)
        # Skip if it looks like a normal word or number
        if key.isdigit() or len(key) < 20:
            return key
        # Show first 4 and last 4 characters for debugging
        if len(key) > 8:
            return f"{key[:4]}{PARTIAL_MASK}{key[-4:]}"
        return REDACTED_MASK
    
    return API_KEY_PATTERN.sub(replace_key, text)


def redact_secrets(text: str) -> str:
    """
    Redact common secret patterns from text
    
    Args:
        text: Input text potentially containing secrets
        
    Returns:
        Text with secrets replaced by [REDACTED]
    """
    result = text
    
    for pattern in SECRET_PATTERNS:
        def replace_secret(match):
            prefix = match.group(1) if match.groups() else ""
            if len(match.groups()) >= 2:
                # Keep the key name but redact the value
                return f"{prefix}={REDACTED_MASK}"
            return REDACTED_MASK
        
        result = pattern.sub(replace_secret, result)
    
    return result


def redact(text: Union[str, Any]) -> str:
    """
    Central redaction function that applies all redaction rules
    
    Args:
        text: Input text or any object (will be converted to string)
        
    Returns:
        Redacted text with sensitive data masked
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Apply all redaction rules in sequence
    text = redact_email(text)
    text = redact_api_keys(text)
    text = redact_secrets(text)
    
    return text


def redact_dict(data: dict) -> dict:
    """
    Recursively redact sensitive data from dictionary values
    
    Args:
        data: Dictionary potentially containing sensitive data
        
    Returns:
        New dictionary with sensitive values redacted
    """
    result = {}
    
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = redact_dict(value)
        elif isinstance(value, list):
            result[key] = [redact(item) if isinstance(item, str) else item for item in value]
        elif isinstance(value, str):
            result[key] = redact(value)
        else:
            result[key] = value
    
    return result


def is_sensitive_field(field_name: str) -> bool:
    """
    Check if a field name suggests sensitive data
    
    Args:
        field_name: Field/key name to check
        
    Returns:
        True if field name suggests sensitive data
    """
    sensitive_fields = {
        'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'auth',
        'authorization', 'credentials', 'api_key', 'access_token',
        'refresh_token', 'private_key', 'cert', 'certificate'
    }
    
    return field_name.lower() in sensitive_fields or any(
        sensitive in field_name.lower() for sensitive in sensitive_fields
    )


def safe_log_format(message: str, **kwargs) -> str:
    """
    Format a log message with automatic redaction of sensitive data
    
    Args:
        message: Base log message
        **kwargs: Additional context data to include
        
    Returns:
        Formatted and redacted log message
    """
    # Redact the base message
    safe_message = redact(message)
    
    # Add context with redaction
    if kwargs:
        safe_context = {}
        for key, value in kwargs.items():
            if is_sensitive_field(key):
                safe_context[key] = REDACTED_MASK
            else:
                safe_context[key] = redact(value) if isinstance(value, str) else value
        
        context_str = ", ".join(f"{k}={v}" for k, v in safe_context.items())
        safe_message = f"{safe_message} [{context_str}]"
    
    return safe_message