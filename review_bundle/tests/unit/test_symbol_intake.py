#!/usr/bin/env python3
"""
Unit tests for symbol intake service
Tests symbol validation, deduplication, and idempotent upserts
"""

import os
import sys
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestSymbolIntakeService:
    """Test symbol intake service functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        self.invalid_symbols = ['123', 'abc', '', None, 'TOOLONG', 'AA@PL']
    
    def test_symbol_validation_valid_formats(self):
        """Test that valid symbol formats are accepted"""
        from services.symbol_intake import SymbolIntakeService
        
        service = SymbolIntakeService()
        
        valid_symbols = [
            'AAPL', 'MSFT', 'BRK.B', 'BF-B', 'GOOGL', 'T', 'GE'
        ]
        
        for symbol in valid_symbols:
            assert service._validate_symbol(symbol), f"Symbol {symbol} should be valid"
            assert service._validate_symbol(symbol.lower()), f"Lowercase {symbol} should be valid"
    
    def test_symbol_validation_invalid_formats(self):
        """Test that invalid symbol formats are rejected"""
        from services.symbol_intake import SymbolIntakeService
        
        service = SymbolIntakeService()
        
        invalid_symbols = [
            '', None, '123', 'abc', 'TOOLONG', 'AA@PL', '   ', 
            'A..B', 'A--B', 'A.B.C', 'AAPL.', 'AAPL-', '.AAPL'
        ]
        
        for symbol in invalid_symbols:
            assert not service._validate_symbol(symbol), f"Symbol {symbol} should be invalid"
    
    def test_intake_symbols_feature_disabled(self):
        """Test symbol intake when feature flag is disabled"""
        with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag:
            mock_flag.return_value = False
            
            from services.symbol_intake import SymbolIntakeService
            
            service = SymbolIntakeService()
            result = service.intake_symbols(['AAPL', 'MSFT'])
            
            assert result['enabled'] is False
            assert result['processed'] == 0
            assert result['accepted'] == 0
            assert result['rejected'] == 0
            assert result['new_symbols'] == []
    
    def test_intake_symbols_empty_candidates(self):
        """Test symbol intake with empty candidate list"""
        with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag:
            mock_flag.return_value = True
            
            from services.symbol_intake import SymbolIntakeService
            
            service = SymbolIntakeService()
            result = service.intake_symbols([])
            
            assert result['enabled'] is True
            assert result['processed'] == 0
            assert result['accepted'] == 0
            assert result['rejected'] == 0
            assert result['new_symbols'] == []
    
    def test_intake_symbols_validation_and_dedup(self):
        """Test symbol intake with validation and deduplication"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            # Create test registry with existing symbols
            existing_df = pd.DataFrame({
                'symbol': ['AAPL', 'MSFT'],
                'source': ['config', 'intake'],
                'added_at': ['2023-01-01T00:00:00Z', '2023-01-02T00:00:00Z']
            })
            existing_df.to_csv(registry_path, index=False)
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', ['TSLA']), \
                 patch('config.tickers.WATCHLIST', []):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import SymbolIntakeService
                
                # Patch the registry path
                service = SymbolIntakeService()
                service.registry_path = Path(registry_path)
                
                # Test candidates: valid new, duplicate, invalid
                candidates = [
                    'GOOGL',  # Valid new
                    'AAPL',   # Duplicate (registry)
                    'TSLA',   # Duplicate (config)
                    'NVDA',   # Valid new
                    'invalid', # Invalid format
                    '',       # Invalid format
                    'AMZ'     # Valid new
                ]
                
                result = service.intake_symbols(candidates)
                
                assert result['enabled'] is True
                assert result['processed'] == 7
                assert result['accepted'] == 3  # GOOGL, NVDA, AMZ
                assert result['rejected'] == 4  # AAPL, TSLA, invalid, empty
                assert set(result['new_symbols']) == {'GOOGL', 'NVDA', 'AMZ'}
                assert len(result['rejected_symbols']) == 4
                
                # Check rejection reasons
                rejection_reasons = [r['reason'] for r in result['rejected_symbols']]
                assert 'duplicate' in rejection_reasons
                assert 'invalid_format' in rejection_reasons
    
    def test_registry_update_atomic_write(self):
        """Test that registry updates use atomic writes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag:
                mock_flag.return_value = True
                
                from services.symbol_intake import SymbolIntakeService
                
                service = SymbolIntakeService()
                service.registry_path = Path(registry_path)
                service._ensure_registry_exists()
                
                # Test atomic write
                new_symbols = ['GOOGL', 'NVDA']
                service._update_registry(new_symbols)
                
                # Verify registry was updated
                assert os.path.exists(registry_path)
                registry_df = pd.read_csv(registry_path)
                
                assert len(registry_df) == 2
                assert set(registry_df['symbol']) == {'GOOGL', 'NVDA'}
                assert all(registry_df['source'] == 'intake')
                assert all(registry_df['added_at'].str.endswith('Z'))  # UTC format
    
    def test_idempotent_upserts(self):
        """Test that multiple runs don't duplicate rows"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', []), \
                 patch('config.tickers.WATCHLIST', []):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import SymbolIntakeService
                
                service = SymbolIntakeService()
                service.registry_path = Path(registry_path)
                
                # First run
                result1 = service.intake_symbols(['GOOGL', 'NVDA'])
                assert result1['accepted'] == 2
                
                # Second run with same symbols
                result2 = service.intake_symbols(['GOOGL', 'NVDA'])
                assert result2['accepted'] == 0
                assert result2['rejected'] == 2
                
                # Verify no duplicates in registry
                registry_df = pd.read_csv(registry_path)
                assert len(registry_df) == 2
                assert set(registry_df['symbol']) == {'GOOGL', 'NVDA'}
    
    def test_deterministic_ordering(self):
        """Test that registry maintains deterministic symbol ordering"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', []), \
                 patch('config.tickers.WATCHLIST', []):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import SymbolIntakeService
                
                service = SymbolIntakeService()
                service.registry_path = Path(registry_path)
                
                # Add symbols in non-alphabetical order
                result = service.intake_symbols(['TSLA', 'AAPL', 'MSFT', 'GOOGL'])
                assert result['accepted'] == 4
                
                # Verify alphabetical ordering in registry
                registry_df = pd.read_csv(registry_path)
                symbols = registry_df['symbol'].tolist()
                assert symbols == ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    def test_get_registry_symbols(self):
        """Test retrieving symbols from registry"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            # Create test registry
            test_df = pd.DataFrame({
                'symbol': ['AAPL', 'MSFT', 'GOOGL'],
                'source': ['intake', 'intake', 'intake'],
                'added_at': ['2023-01-01T00:00:00Z'] * 3
            })
            test_df.to_csv(registry_path, index=False)
            
            from services.symbol_intake import SymbolIntakeService
            
            service = SymbolIntakeService()
            service.registry_path = Path(registry_path)
            
            symbols = service.get_registry_symbols()
            assert set(symbols) == {'AAPL', 'MSFT', 'GOOGL'}
    
    def test_get_registry_symbols_empty_registry(self):
        """Test retrieving symbols from empty registry"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            from services.symbol_intake import SymbolIntakeService
            
            service = SymbolIntakeService()
            service.registry_path = Path(registry_path)
            
            symbols = service.get_registry_symbols()
            assert symbols == []
    
    def test_existing_symbols_deduplication(self):
        """Test deduplication against existing config symbols"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', ['AAPL', 'MSFT']), \
                 patch('config.tickers.WATCHLIST', ['GOOGL']):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import SymbolIntakeService
                
                service = SymbolIntakeService()
                service.registry_path = Path(registry_path)
                
                # Try to add symbols that exist in config
                result = service.intake_symbols(['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
                
                assert result['accepted'] == 1  # Only TSLA should be accepted
                assert result['rejected'] == 3  # AAPL, MSFT, GOOGL are duplicates
                assert result['new_symbols'] == ['TSLA']
    
    def test_case_insensitive_processing(self):
        """Test that symbols are processed case-insensitively"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', ['AAPL']), \
                 patch('config.tickers.WATCHLIST', []):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import SymbolIntakeService
                
                service = SymbolIntakeService()
                service.registry_path = Path(registry_path)
                
                # Try to add lowercase version of existing symbol
                result = service.intake_symbols(['aapl', 'MSFT', 'googl'])
                
                assert result['accepted'] == 2  # MSFT, GOOGL
                assert result['rejected'] == 1  # aapl is duplicate
                
                # Verify symbols are stored as uppercase
                registry_df = pd.read_csv(registry_path)
                assert set(registry_df['symbol']) == {'GOOGL', 'MSFT'}
    
    def test_rejection_reason_breakdown(self):
        """Test that rejection reasons are properly categorized"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', ['AAPL']), \
                 patch('config.tickers.WATCHLIST', []):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import SymbolIntakeService
                
                service = SymbolIntakeService()
                service.registry_path = Path(registry_path)
                
                candidates = [
                    'AAPL',      # duplicate
                    'invalid',   # invalid_format
                    '',          # invalid_format
                    None,        # invalid_type
                    123,         # invalid_type
                    'MSFT'       # valid, should be accepted
                ]
                
                result = service.intake_symbols(candidates)
                
                # Check rejection breakdown
                rejected = result['rejected_symbols']
                reasons = [r['reason'] for r in rejected]
                
                assert reasons.count('duplicate') == 1
                assert reasons.count('invalid_format') == 2
                assert reasons.count('invalid_type') == 2
                
                assert result['accepted'] == 1
                assert result['new_symbols'] == ['MSFT']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])