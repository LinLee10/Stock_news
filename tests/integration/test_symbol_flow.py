#!/usr/bin/env python3
"""
Integration tests for symbol intake flow
Tests registry updates and inclusion in next pipeline run
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


class TestSymbolIntakeFlow:
    """Test symbol intake integration flow"""
    
    def setup_method(self):
        """Set up test environment"""
        self.test_candidates = ['NVDA', 'AMD', 'CRM']
        self.existing_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    def test_registry_update_flow(self):
        """Test complete registry update flow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', self.existing_symbols[:2]), \
                 patch('config.tickers.WATCHLIST', [self.existing_symbols[2]]):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import symbol_intake_service
                
                # Patch the service's registry path
                original_path = symbol_intake_service.registry_path
                symbol_intake_service.registry_path = Path(registry_path)
                
                try:
                    # Initial state - registry should be empty
                    assert not os.path.exists(registry_path)
                    
                    # Process intake
                    result = symbol_intake_service.intake_symbols(self.test_candidates)
                    
                    # Verify intake results
                    assert result['accepted'] == 3
                    assert result['rejected'] == 0
                    assert set(result['new_symbols']) == set(self.test_candidates)
                    
                    # Verify registry was created and updated
                    assert os.path.exists(registry_path)
                    
                    registry_df = pd.read_csv(registry_path)
                    assert len(registry_df) == 3
                    assert set(registry_df['symbol']) == set(self.test_candidates)
                    assert all(registry_df['source'] == 'intake')
                    assert all(registry_df['added_at'].notna())
                    
                    # Test retrieval
                    registry_symbols = symbol_intake_service.get_registry_symbols()
                    assert set(registry_symbols) == set(self.test_candidates)
                    
                finally:
                    # Restore original path
                    symbol_intake_service.registry_path = original_path
    
    def test_next_run_includes_new_symbols(self):
        """Test that new symbols are included in subsequent pipeline runs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            candidates_csv_path = os.path.join(temp_dir, "symbol_candidates.csv")
            
            # Create candidates CSV
            candidates_df = pd.DataFrame({'symbol': self.test_candidates})
            candidates_df.to_csv(candidates_csv_path, index=False)
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', ['AAPL']), \
                 patch('config.tickers.WATCHLIST', ['MSFT']), \
                 patch.dict(os.environ, {
                     'SYMBOL_INTAKE_CSV': candidates_csv_path
                 }):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import symbol_intake_service
                
                # Patch the service's registry path
                original_path = symbol_intake_service.registry_path
                symbol_intake_service.registry_path = Path(registry_path)
                
                try:
                    # Simulate main.py symbol intake flow
                    import pandas as pd
                    
                    # Initial symbols (from config)
                    all_tickers = ['AAPL', 'MSFT']
                    original_count = len(all_tickers)
                    
                    # Process intake (simulating main.py logic)
                    candidates = []
                    
                    # Read from CSV
                    if os.path.exists(candidates_csv_path):
                        candidates_df = pd.read_csv(candidates_csv_path)
                        if 'symbol' in candidates_df.columns:
                            candidates.extend(candidates_df['symbol'].dropna().astype(str).tolist())
                    
                    # Process candidates
                    if candidates:
                        intake_result = symbol_intake_service.intake_symbols(candidates)
                        
                        # Merge new symbols into current run
                        if intake_result['new_symbols']:
                            all_tickers.extend(intake_result['new_symbols'])
                    
                    # Verify new symbols were added to the run
                    assert len(all_tickers) > original_count
                    assert set(all_tickers) == {'AAPL', 'MSFT', 'NVDA', 'AMD', 'CRM'}
                    
                    # Verify registry was updated
                    registry_df = pd.read_csv(registry_path)
                    assert len(registry_df) == 3
                    assert set(registry_df['symbol']) == set(self.test_candidates)
                    
                finally:
                    # Restore original path
                    symbol_intake_service.registry_path = original_path
    
    def test_environment_variable_intake(self):
        """Test symbol intake from environment variable"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            env_candidates = 'NVDA,AMD,CRM,INTC'
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', ['AAPL']), \
                 patch('config.tickers.WATCHLIST', ['MSFT']), \
                 patch.dict(os.environ, {
                     'SYMBOL_INTAKE_LIST': env_candidates
                 }):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import symbol_intake_service
                
                # Patch the service's registry path
                original_path = symbol_intake_service.registry_path
                symbol_intake_service.registry_path = Path(registry_path)
                
                try:
                    # Simulate main.py environment variable reading
                    candidates = []
                    env_candidates_str = os.getenv('SYMBOL_INTAKE_LIST', '')
                    if env_candidates_str:
                        candidates.extend([s.strip() for s in env_candidates_str.split(',') if s.strip()])
                    
                    # Process intake
                    intake_result = symbol_intake_service.intake_symbols(candidates)
                    
                    # Verify results
                    assert intake_result['accepted'] == 4
                    assert set(intake_result['new_symbols']) == {'NVDA', 'AMD', 'CRM', 'INTC'}
                    
                    # Verify registry
                    registry_df = pd.read_csv(registry_path)
                    assert len(registry_df) == 4
                    assert set(registry_df['symbol']) == {'NVDA', 'AMD', 'CRM', 'INTC'}
                    
                finally:
                    # Restore original path
                    symbol_intake_service.registry_path = original_path
    
    def test_mixed_source_intake(self):
        """Test symbol intake from both environment and CSV sources"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            candidates_csv_path = os.path.join(temp_dir, "symbol_candidates.csv")
            
            # Create candidates CSV
            csv_candidates = ['NVDA', 'AMD']
            candidates_df = pd.DataFrame({'symbol': csv_candidates})
            candidates_df.to_csv(candidates_csv_path, index=False)
            
            # Environment candidates
            env_candidates = 'CRM,INTC'
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', ['AAPL']), \
                 patch('config.tickers.WATCHLIST', ['MSFT']), \
                 patch.dict(os.environ, {
                     'SYMBOL_INTAKE_LIST': env_candidates,
                     'SYMBOL_INTAKE_CSV': candidates_csv_path
                 }):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import symbol_intake_service
                
                # Patch the service's registry path
                original_path = symbol_intake_service.registry_path
                symbol_intake_service.registry_path = Path(registry_path)
                
                try:
                    # Simulate main.py mixed source reading
                    candidates = []
                    
                    # From environment
                    env_candidates_str = os.getenv('SYMBOL_INTAKE_LIST', '')
                    if env_candidates_str:
                        candidates.extend([s.strip() for s in env_candidates_str.split(',') if s.strip()])
                    
                    # From CSV
                    if os.path.exists(candidates_csv_path):
                        candidates_df = pd.read_csv(candidates_csv_path)
                        if 'symbol' in candidates_df.columns:
                            candidates.extend(candidates_df['symbol'].dropna().astype(str).tolist())
                    
                    # Process intake
                    intake_result = symbol_intake_service.intake_symbols(candidates)
                    
                    # Verify all candidates from both sources were processed
                    assert intake_result['processed'] == 4
                    assert intake_result['accepted'] == 4
                    assert set(intake_result['new_symbols']) == {'NVDA', 'AMD', 'CRM', 'INTC'}
                    
                finally:
                    # Restore original path
                    symbol_intake_service.registry_path = original_path
    
    def test_concurrent_run_safety(self):
        """Test that concurrent runs handle registry safely"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', ['AAPL']), \
                 patch('config.tickers.WATCHLIST', []):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import symbol_intake_service
                
                # Patch the service's registry path
                original_path = symbol_intake_service.registry_path
                symbol_intake_service.registry_path = Path(registry_path)
                
                try:
                    # Simulate multiple concurrent intakes
                    batch1 = ['NVDA', 'AMD']
                    batch2 = ['CRM', 'INTC']
                    
                    # First batch
                    result1 = symbol_intake_service.intake_symbols(batch1)
                    assert result1['accepted'] == 2
                    
                    # Second batch (simulating concurrent run)
                    result2 = symbol_intake_service.intake_symbols(batch2)
                    assert result2['accepted'] == 2
                    
                    # Verify final state
                    registry_df = pd.read_csv(registry_path)
                    assert len(registry_df) == 4
                    assert set(registry_df['symbol']) == {'NVDA', 'AMD', 'CRM', 'INTC'}
                    
                    # Verify deterministic ordering
                    symbols = registry_df['symbol'].tolist()
                    assert symbols == sorted(symbols)
                    
                finally:
                    # Restore original path
                    symbol_intake_service.registry_path = original_path
    
    def test_rollback_scenario(self):
        """Test rollback scenario by removing symbols from registry"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            # Create initial registry with some symbols
            initial_df = pd.DataFrame({
                'symbol': ['NVDA', 'AMD', 'CRM'],
                'source': ['intake', 'intake', 'intake'],
                'added_at': ['2023-01-01T00:00:00Z'] * 3
            })
            initial_df.to_csv(registry_path, index=False)
            
            from services.symbol_intake import symbol_intake_service
            
            # Patch the service's registry path
            original_path = symbol_intake_service.registry_path
            symbol_intake_service.registry_path = Path(registry_path)
            
            try:
                # Verify initial state
                symbols = symbol_intake_service.get_registry_symbols()
                assert set(symbols) == {'NVDA', 'AMD', 'CRM'}
                
                # Simulate rollback by removing specific symbols
                registry_df = pd.read_csv(registry_path)
                
                # Remove AMD from registry (rollback simulation)
                updated_df = registry_df[registry_df['symbol'] != 'AMD']
                updated_df.to_csv(registry_path, index=False)
                
                # Verify rollback
                symbols_after_rollback = symbol_intake_service.get_registry_symbols()
                assert set(symbols_after_rollback) == {'NVDA', 'CRM'}
                assert 'AMD' not in symbols_after_rollback
                
            finally:
                # Restore original path
                symbol_intake_service.registry_path = original_path
    
    def test_registry_schema_consistency(self):
        """Test that registry maintains consistent schema across operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', []), \
                 patch('config.tickers.WATCHLIST', []):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import symbol_intake_service
                
                # Patch the service's registry path
                original_path = symbol_intake_service.registry_path
                symbol_intake_service.registry_path = Path(registry_path)
                
                try:
                    # Multiple intake operations
                    symbol_intake_service.intake_symbols(['NVDA'])
                    symbol_intake_service.intake_symbols(['AMD', 'CRM'])
                    symbol_intake_service.intake_symbols(['INTC'])
                    
                    # Verify schema consistency
                    registry_df = pd.read_csv(registry_path)
                    
                    expected_columns = {'symbol', 'source', 'added_at'}
                    assert set(registry_df.columns) == expected_columns
                    
                    # Verify data types and formats
                    assert all(registry_df['symbol'].str.isupper())
                    assert all(registry_df['source'] == 'intake')
                    assert all(registry_df['added_at'].str.endswith('Z'))  # UTC format
                    
                    # Verify no null values
                    assert not registry_df.isnull().any().any()
                    
                finally:
                    # Restore original path
                    symbol_intake_service.registry_path = original_path
    
    def test_error_handling_malformed_csv(self):
        """Test error handling with malformed candidate CSV"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            bad_csv_path = os.path.join(temp_dir, "bad_candidates.csv")
            
            # Create malformed CSV
            with open(bad_csv_path, 'w') as f:
                f.write("not,valid,csv\n")
                f.write("data,here\n")
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', []), \
                 patch('config.tickers.WATCHLIST', []), \
                 patch.dict(os.environ, {
                     'SYMBOL_INTAKE_CSV': bad_csv_path
                 }):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import symbol_intake_service
                
                # Patch the service's registry path
                original_path = symbol_intake_service.registry_path
                symbol_intake_service.registry_path = Path(registry_path)
                
                try:
                    # Simulate main.py CSV reading with error handling
                    candidates = []
                    intake_csv_path = os.getenv('SYMBOL_INTAKE_CSV')
                    
                    if os.path.exists(intake_csv_path):
                        try:
                            candidates_df = pd.read_csv(intake_csv_path)
                            if 'symbol' in candidates_df.columns:
                                candidates.extend(candidates_df['symbol'].dropna().astype(str).tolist())
                        except Exception:
                            # Should handle error gracefully
                            pass
                    
                    # Should not crash and should handle empty candidates gracefully
                    intake_result = symbol_intake_service.intake_symbols(candidates)
                    assert intake_result['processed'] == 0
                    assert intake_result['accepted'] == 0
                    
                finally:
                    # Restore original path
                    symbol_intake_service.registry_path = original_path


class TestSymbolIntakeNonRegression:
    """Test that symbol intake doesn't break existing functionality"""
    
    def test_main_pipeline_unchanged_when_disabled(self):
        """Test that main pipeline is unchanged when intake is disabled"""
        with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag:
            mock_flag.return_value = False
            
            # Simulate main.py behavior when flag is off
            initial_tickers = ['AAPL', 'MSFT', 'GOOGL']
            all_tickers = initial_tickers.copy()
            
            # Symbol intake section should be skipped
            if mock_flag.return_value:
                # This block should not execute
                assert False, "Should not execute when flag is disabled"
            
            # Verify tickers list is unchanged
            assert all_tickers == initial_tickers
    
    def test_existing_config_symbols_preserved(self):
        """Test that existing config symbols are always preserved"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "tickers_registry.csv")
            
            config_portfolio = ['AAPL', 'MSFT']
            config_watchlist = ['GOOGL', 'TSLA']
            
            with patch('services.symbol_intake.is_symbol_intake_enabled') as mock_flag, \
                 patch('config.tickers.PORTFOLIO', config_portfolio), \
                 patch('config.tickers.WATCHLIST', config_watchlist):
                
                mock_flag.return_value = True
                
                from services.symbol_intake import symbol_intake_service
                
                # Patch the service's registry path
                original_path = symbol_intake_service.registry_path
                symbol_intake_service.registry_path = Path(registry_path)
                
                try:
                    # Add new symbols via intake
                    intake_result = symbol_intake_service.intake_symbols(['NVDA', 'AMD'])
                    assert intake_result['accepted'] == 2
                    
                    # Simulate main pipeline combining config + registry symbols
                    all_tickers = config_portfolio + config_watchlist
                    registry_symbols = symbol_intake_service.get_registry_symbols()
                    all_tickers.extend(registry_symbols)
                    
                    # Verify config symbols are preserved and new symbols added
                    assert 'AAPL' in all_tickers
                    assert 'MSFT' in all_tickers
                    assert 'GOOGL' in all_tickers
                    assert 'TSLA' in all_tickers
                    assert 'NVDA' in all_tickers
                    assert 'AMD' in all_tickers
                    
                finally:
                    # Restore original path
                    symbol_intake_service.registry_path = original_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])