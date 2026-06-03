#!/usr/bin/env python3
"""
Symbol intake service for managing new ticker onboarding
Handles symbol validation, deduplication, and record management
"""
import pandas as pd
import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from config.feature_flags import is_symbol_intake_enabled

logger = logging.getLogger(__name__)

# BEGIN F17_IMPL
class SymbolIntakeService:
    """Service for managing symbol intake and registry"""
    
    def __init__(self):
        self.data_path = Path("data")
        self.registry_path = self.data_path / "tickers_registry.csv"
        self._ensure_registry_exists()
    
    def _ensure_registry_exists(self):
        """Ensure tickers registry file exists with correct schema"""
        if not self.registry_path.exists():
            self.data_path.mkdir(exist_ok=True)
            # Create empty registry with F17 schema: symbol,source,added_at
            empty_registry = pd.DataFrame(columns=['symbol', 'source', 'added_at'])
            empty_registry.to_csv(self.registry_path, index=False)
            logger.info(f"Created empty tickers registry at {self.registry_path}")
    
    def _validate_symbol(self, symbol: str) -> bool:
        """
        Validate ticker symbol format
        Allows A-Z, dot, hyphen patterns
        
        Args:
            symbol: Ticker symbol to validate
            
        Returns:
            True if valid format, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Allow uppercase letters, dots, hyphens
        # Examples: AAPL, BRK.B, BF-B
        pattern = r'^[A-Z]{1,5}([.-][A-Z]{1,2})?$'
        return bool(re.match(pattern, symbol.upper().strip()))
    
    def _get_existing_symbols(self) -> set:
        """Get all existing symbols from registry and config"""
        existing_symbols = set()
        
        # Get symbols from registry
        try:
            if self.registry_path.exists():
                registry_df = pd.read_csv(self.registry_path)
                if 'symbol' in registry_df.columns:
                    existing_symbols.update(registry_df['symbol'].str.upper().tolist())
        except Exception as e:
            logger.warning(f"Error reading registry: {e}")
        
        # Get symbols from config (PORTFOLIO, WATCHLIST)
        try:
            from config.tickers import PORTFOLIO, WATCHLIST
            existing_symbols.update([s.upper() for s in PORTFOLIO])
            existing_symbols.update([s.upper() for s in WATCHLIST])
        except ImportError:
            logger.warning("Could not import config tickers for deduplication")
        
        return existing_symbols
    
    def intake_symbols(self, candidates: List[str]) -> Dict[str, Any]:
        """
        Batch intake of new ticker symbols with validation and deduplication
        
        Args:
            candidates: List of candidate ticker symbols to intake
            
        Returns:
            Dictionary with intake summary and results
        """
        if not is_symbol_intake_enabled():
            logger.debug("Symbol intake feature disabled")
            return {
                'enabled': False,
                'processed': 0,
                'accepted': 0,
                'rejected': 0,
                'new_symbols': []
            }
        
        if not candidates:
            logger.info("No candidate symbols provided for intake")
            return {
                'enabled': True,
                'processed': 0,
                'accepted': 0,
                'rejected': 0,
                'new_symbols': []
            }
        
        start_time = datetime.now()
        logger.info(f"F17: Starting symbol intake for {len(candidates)} candidates")
        
        # Get existing symbols for deduplication
        existing_symbols = self._get_existing_symbols()
        
        # Process candidates
        accepted_symbols = []
        rejected_symbols = []
        
        for candidate in candidates:
            if not candidate or not isinstance(candidate, str):
                rejected_symbols.append({'symbol': str(candidate), 'reason': 'invalid_type'})
                continue
            
            symbol = candidate.strip().upper()
            
            # Validate format
            if not self._validate_symbol(symbol):
                rejected_symbols.append({'symbol': symbol, 'reason': 'invalid_format'})
                continue
            
            # Check for duplicates
            if symbol in existing_symbols:
                rejected_symbols.append({'symbol': symbol, 'reason': 'duplicate'})
                continue
            
            # Accept symbol
            accepted_symbols.append(symbol)
            existing_symbols.add(symbol)  # Prevent duplicates within this batch
        
        # Update registry with accepted symbols
        if accepted_symbols:
            self._update_registry(accepted_symbols)
        
        # Log summary
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = {
            'enabled': True,
            'processed': len(candidates),
            'accepted': len(accepted_symbols),
            'rejected': len(rejected_symbols),
            'new_symbols': accepted_symbols,
            'rejected_symbols': rejected_symbols,
            'processing_time_seconds': processing_time
        }
        
        logger.info(f"F17: Symbol intake completed - {summary['accepted']} accepted, "
                   f"{summary['rejected']} rejected, {processing_time:.2f}s")
        
        # Log individual rejections for debugging
        if rejected_symbols:
            rejection_summary = {}
            for rejection in rejected_symbols:
                reason = rejection['reason']
                rejection_summary[reason] = rejection_summary.get(reason, 0) + 1
            logger.info(f"F17: Rejection breakdown: {dict(rejection_summary)}")
        
        return summary
    
    def _update_registry(self, new_symbols: List[str]):
        """
        Update registry with new symbols using atomic write
        
        Args:
            new_symbols: List of validated symbols to add
        """
        try:
            # Read existing registry
            if self.registry_path.exists():
                registry_df = pd.read_csv(self.registry_path)
            else:
                registry_df = pd.DataFrame(columns=['symbol', 'source', 'added_at'])
            
            # Create new records
            timestamp = datetime.utcnow().isoformat() + 'Z'
            new_records = []
            
            for symbol in new_symbols:
                new_records.append({
                    'symbol': symbol,
                    'source': 'intake',
                    'added_at': timestamp
                })
            
            # Append new records
            if new_records:
                new_df = pd.DataFrame(new_records)
                updated_registry = pd.concat([registry_df, new_df], ignore_index=True)
                
                # Sort by symbol for deterministic ordering
                updated_registry = updated_registry.sort_values('symbol').reset_index(drop=True)
                
                # Atomic write using temp file
                temp_path = f"{self.registry_path}.tmp"
                updated_registry.to_csv(temp_path, index=False)
                os.rename(temp_path, self.registry_path)
                
                logger.info(f"F17: Updated registry with {len(new_symbols)} new symbols")
        
        except Exception as e:
            logger.error(f"F17: Failed to update registry: {e}")
            raise
    
    def get_registry_symbols(self) -> List[str]:
        """
        Get all symbols from the registry
        
        Returns:
            List of symbols from registry
        """
        try:
            if not self.registry_path.exists():
                return []
            
            registry_df = pd.read_csv(self.registry_path)
            if 'symbol' in registry_df.columns:
                return registry_df['symbol'].tolist()
            else:
                return []
        
        except Exception as e:
            logger.error(f"Error reading registry symbols: {e}")
            return []


# Global instance
symbol_intake_service = SymbolIntakeService()
# END F17_IMPL