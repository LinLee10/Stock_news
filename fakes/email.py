#!/usr/bin/env python3
"""Fake email sender for DRY_RUN mode"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from settings import EMAIL_ARTIFACTS_DIR


class FakeEmailSender:
    """Fake email sender that writes to files instead of network"""
    
    def __init__(self):
        self.sent_emails = []
        self.call_count = 0
        # Ensure artifacts directory exists
        Path(EMAIL_ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    
    def send_email(self, 
                   subject: str, 
                   body: str, 
                   to_email: str = None, 
                   html_body: str = None,
                   attachments: List = None,
                   **kwargs) -> Dict[str, Any]:
        """Write email to file instead of sending"""
        self.call_count += 1
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]  # Include microseconds
        safe_subject = "".join(c for c in subject if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        safe_subject = safe_subject.replace(' ', '_')
        
        # Write HTML version
        if html_body:
            html_path = Path(EMAIL_ARTIFACTS_DIR) / f"{timestamp}_{safe_subject}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(f"<!-- Subject: {subject} -->\n")
                f.write(f"<!-- To: {to_email or 'unknown@example.com'} -->\n")
                f.write(f"<!-- Generated: {datetime.now().isoformat()} -->\n")
                f.write(f"<!-- DRY_RUN MODE - No actual email sent -->\n\n")
                f.write(html_body)
        
        # Write text version  
        txt_path = Path(EMAIL_ARTIFACTS_DIR) / f"{timestamp}_{safe_subject}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Subject: {subject}\n")
            f.write(f"To: {to_email or 'unknown@example.com'}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("DRY_RUN MODE - No actual email sent\n")
            f.write("-" * 50 + "\n\n")
            f.write(body)
            
            if attachments:
                f.write(f"\n\nAttachments ({len(attachments)}):\n")
                for i, att in enumerate(attachments):
                    f.write(f"  {i+1}. {att}\n")
        
        # Track sent email
        email_record = {
            'subject': subject,
            'to': to_email,
            'timestamp': datetime.now().isoformat(),
            'html_file': html_path if html_body else None,
            'txt_file': txt_path,
            'has_attachments': bool(attachments)
        }
        self.sent_emails.append(email_record)
        
        print(f"📧 DRY_RUN: Email written to {txt_path}")
        return {
            'success': True, 
            'message': f'Email written to {txt_path}',
            'file_path': str(txt_path)
        }
    
    async def send_email_async(self, *args, **kwargs):
        """Async version of send_email"""
        return self.send_email(*args, **kwargs)


# Standalone functions for backward compatibility
def fake_send_email(subject: str, body: str, to_email: str = None, html_body: str = None, **kwargs):
    """Write email to file instead of sending"""
    sender = FakeEmailSender()
    return sender.send_email(subject, body, to_email, html_body, **kwargs)


async def fake_send_email_async(subject: str, body: str, to_email: str = None, html_body: str = None, **kwargs):
    """Async version"""
    return fake_send_email(subject, body, to_email, html_body, **kwargs)


# Gmail-specific fake
class FakeGmailSender(FakeEmailSender):
    """Fake Gmail sender"""
    
    def __init__(self, username=None, password=None):
        super().__init__()
        self.username = username or "fake@gmail.com"
        self.password = "fake_password"
    
    def login(self):
        """Fake login"""
        print(f"📧 DRY_RUN: Fake Gmail login for {self.username}")
        return True
    
    def logout(self):
        """Fake logout"""
        print("📧 DRY_RUN: Fake Gmail logout")
        return True