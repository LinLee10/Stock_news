#!/usr/bin/env python3
"""Structlog compatibility shim for DRY_RUN mode"""
import logging
import sys
from typing import Any, Dict

class StructlogShim:
    """Shim that provides structlog-like interface using standard logging"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            # Add a handler if none exists
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, msg: str, **kwargs):
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"{msg} {extra_info}" if extra_info else msg)
    
    def warning(self, msg: str, **kwargs):
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.warning(f"{msg} {extra_info}" if extra_info else msg)
    
    def warn(self, msg: str, **kwargs):
        """Alias for warning"""
        self.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.error(f"{msg} {extra_info}" if extra_info else msg)
    
    def debug(self, msg: str, **kwargs):
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"{msg} {extra_info}" if extra_info else msg)
    
    def critical(self, msg: str, **kwargs):
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.critical(f"{msg} {extra_info}" if extra_info else msg)
    
    def bind(self, **kwargs):
        """Return self for chaining (structlog compatibility)"""
        return self
    
    def with_fields(self, **kwargs):
        """Return self for chaining (structlog compatibility)"""
        return self

def get_logger(name: str = None) -> StructlogShim:
    """Return structlog-compatible logger shim"""
    if name is None:
        # Get caller's module name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'unknown')
    return StructlogShim(name)

# Mock structlog module for import compatibility
class FakeStructlog:
    @staticmethod
    def get_logger(name=None):
        return get_logger(name)
    
    @staticmethod
    def configure(*args, **kwargs):
        pass
    
    class processors:
        @staticmethod
        def JSONRenderer():
            return lambda x: str(x)
        
        @staticmethod
        def TimeStamper():
            return lambda x: x

# For direct imports like "import structlog"
structlog = FakeStructlog()