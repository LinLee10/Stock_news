#!/usr/bin/env python3
"""
YFinance Daily Refresh Guard

Implements once-per-day refresh logic with manifest tracking and lockfile protection.
Prevents multiple refreshes on the same day and handles rate limiting gracefully.
"""

import os
import json
import hashlib
import fcntl
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class RefreshManifest:
    """Daily refresh manifest to track state and prevent duplicate refreshes."""
    date_utc: str
    window_hour: int
    key: str
    symbols_hash: str
    status: str  # pending|done|failed
    attempts: int
    symbols: List[str]
    last_update: str
    success_count: int = 0
    failed_count: int = 0


class YFDailyRefreshGuard:
    """
    Guards yfinance refresh to run once per calendar day with proper locking.
    
    Key behaviors:
    - Only runs during configured UTC window 
    - Uses lockfile to prevent concurrent refreshes
    - Tracks state in manifest.json
    - Falls back to cached data on rate limits
    """
    
    def __init__(self, cache_dir: Path = None, config: Dict[str, Any] = None):
        self.cache_dir = cache_dir or Path("data/yf_bulk_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config with defaults
        if config is None:
            from config.config import (YF_REFRESH_WINDOW_UTC_HOUR, YF_DAILY_KEY, 
                                     YF_MAX_RETRIES, YF_BACKOFF_BASE_SECONDS)
            config = {
                'window_hour': YF_REFRESH_WINDOW_UTC_HOUR,
                'daily_key': YF_DAILY_KEY,
                'max_retries': YF_MAX_RETRIES,
                'backoff_base': YF_BACKOFF_BASE_SECONDS
            }
        
        self.config = config
        self.manifest_path = self.cache_dir / "manifest.json"
        self.lock_path = self.cache_dir / ".refresh.lock"
        
    def _compute_symbols_hash(self, symbols: List[str]) -> str:
        """Compute hash of sorted symbols for change detection."""
        symbols_str = "|".join(sorted(symbols))
        return hashlib.sha256(symbols_str.encode()).hexdigest()[:16]
    
    def _load_manifest(self) -> Optional[RefreshManifest]:
        """Load existing manifest if present."""
        if not self.manifest_path.exists():
            return None
            
        try:
            with open(self.manifest_path, 'r') as f:
                data = json.load(f)
                return RefreshManifest(**data)
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")
            return None
    
    def _save_manifest(self, manifest: RefreshManifest):
        """Save manifest atomically."""
        try:
            # Atomic write using temp file
            temp_path = self.manifest_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(asdict(manifest), f, indent=2)
            
            # Atomic rename
            temp_path.rename(self.manifest_path)
            logger.debug(f"Saved manifest: {manifest.status} for {len(manifest.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
    
    def _is_today_utc(self, date_str: str) -> bool:
        """Check if date string matches today in UTC."""
        try:
            manifest_date = datetime.fromisoformat(date_str).date()
            today = datetime.now(timezone.utc).date()
            return manifest_date == today
        except Exception:
            return False
    
    def _is_within_window(self) -> bool:
        """Check if current UTC time is within refresh window."""
        now_utc = datetime.now(timezone.utc)
        return now_utc.hour >= self.config['window_hour']
    
    def _should_refresh(self, symbols: List[str]) -> tuple[bool, Optional[RefreshManifest]]:
        """
        Determine if refresh should run based on manifest state.
        
        Returns:
            (should_refresh: bool, existing_manifest: Optional[RefreshManifest])
        """
        manifest = self._load_manifest()
        
        # No manifest = first run
        if manifest is None:
            if not self._is_within_window():
                logger.info(f"YF refresh deferred: before window hour {self.config['window_hour']} UTC")
                return False, None
            logger.info("YF refresh: no manifest found, will create new one")
            return True, None
        
        # Check if same day
        if not self._is_today_utc(manifest.date_utc):
            if not self._is_within_window():
                logger.info(f"YF refresh deferred: before window hour {self.config['window_hour']} UTC")
                return False, manifest
            logger.info("YF refresh: new day detected")
            return True, manifest
        
        # Same day - check if symbols/config changed
        symbols_hash = self._compute_symbols_hash(symbols)
        if (manifest.symbols_hash != symbols_hash or 
            manifest.key != self.config['daily_key']):
            logger.info("YF refresh: symbols or config changed")
            return True, manifest
        
        # Same day, same config - check status
        if manifest.status == "done":
            logger.info(f"YF refresh: already completed today for {len(symbols)} symbols")
            return False, manifest
        elif manifest.status == "failed" and manifest.attempts >= self.config['max_retries']:
            logger.info("YF refresh: failed today, max attempts reached")
            return False, manifest
        else:
            logger.info(f"YF refresh: retrying (status={manifest.status}, attempts={manifest.attempts})")
            return True, manifest
    
    def _acquire_lock(self, timeout: int = 30) -> Optional[int]:
        """Acquire refresh lock with timeout."""
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            
            # Try to acquire lock
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    logger.debug("Acquired YF refresh lock")
                    return fd
                except IOError:
                    time.sleep(0.1)
            
            # Timeout
            os.close(fd)
            os.unlink(self.lock_path)
            logger.warning("YF refresh lock timeout")
            return None
            
        except FileExistsError:
            logger.info("YF refresh already in progress (lock exists)")
            return None
        except Exception as e:
            logger.error(f"Failed to acquire YF refresh lock: {e}")
            return None
    
    def _release_lock(self, fd: int):
        """Release refresh lock."""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            os.unlink(self.lock_path)
            logger.debug("Released YF refresh lock")
        except Exception as e:
            logger.error(f"Failed to release YF refresh lock: {e}")
    
    def _perform_refresh(self, symbols: List[str], existing_manifest: Optional[RefreshManifest]) -> RefreshManifest:
        """Perform the actual refresh with bounded retries."""
        now = datetime.now(timezone.utc)
        symbols_hash = self._compute_symbols_hash(symbols)
        
        # Create new manifest or update existing
        if existing_manifest and existing_manifest.status == "pending":
            # Continue existing attempt
            manifest = existing_manifest
            manifest.attempts += 1
            manifest.last_update = now.isoformat()
        else:
            # New attempt
            manifest = RefreshManifest(
                date_utc=now.date().isoformat(),
                window_hour=self.config['window_hour'],
                key=self.config['daily_key'],
                symbols_hash=symbols_hash,
                status="pending",
                attempts=1,
                symbols=symbols,
                last_update=now.isoformat(),
                success_count=0,
                failed_count=0
            )
        
        # Save pending status
        self._save_manifest(manifest)
        
        # Attempt refresh
        logger.info(f"YF refresh: starting attempt {manifest.attempts} for {len(symbols)} symbols")
        
        try:
            from services.data_sources.yfinance_provider import create_yfinance_provider
            provider = create_yfinance_provider()
            
            # Fetch data with bounded retries
            results = provider.fetch_history(symbols)
            
            # Update counters
            manifest.success_count = len(results)
            manifest.failed_count = len(symbols) - len(results)
            
            if results:
                manifest.status = "done"
                logger.info(f"YF refresh: completed {manifest.success_count}/{len(symbols)} symbols")
            else:
                manifest.status = "failed"
                logger.warning(f"YF refresh: all symbols failed (likely rate limited)")
            
        except Exception as e:
            manifest.status = "failed"
            manifest.failed_count = len(symbols)
            logger.error(f"YF refresh: failed with exception: {e}")
        
        # Update final status
        manifest.last_update = datetime.now(timezone.utc).isoformat()
        self._save_manifest(manifest)
        
        return manifest
    
    def run_once_per_day(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Main entry point: run refresh once per day with proper guards.
        
        Returns:
            Status dict with refresh results
        """
        if not symbols:
            logger.warning("YF refresh: no symbols provided")
            return {"status": "skipped", "reason": "no_symbols"}
        
        # Check if refresh should run
        should_refresh, existing_manifest = self._should_refresh(symbols)
        
        if not should_refresh:
            return {
                "status": "skipped", 
                "reason": "already_done" if existing_manifest and existing_manifest.status == "done" else "deferred",
                "cache_available": existing_manifest is not None
            }
        
        # Acquire lock
        lock_fd = self._acquire_lock()
        if lock_fd is None:
            return {"status": "skipped", "reason": "lock_failed"}
        
        try:
            # Perform refresh
            manifest = self._perform_refresh(symbols, existing_manifest)
            
            return {
                "status": manifest.status,
                "attempts": manifest.attempts,
                "success_count": manifest.success_count,
                "failed_count": manifest.failed_count,
                "symbols_requested": len(symbols)
            }
            
        finally:
            self._release_lock(lock_fd)
    
    @classmethod
    def run_once_per_day_static(cls, symbols: List[str], key: str = None) -> Dict[str, Any]:
        """Static method for easy integration."""
        config = None
        if key:
            from config.config import (YF_REFRESH_WINDOW_UTC_HOUR, YF_MAX_RETRIES, 
                                     YF_BACKOFF_BASE_SECONDS)
            config = {
                'window_hour': YF_REFRESH_WINDOW_UTC_HOUR,
                'daily_key': key,
                'max_retries': YF_MAX_RETRIES,
                'backoff_base': YF_BACKOFF_BASE_SECONDS
            }
        
        guard = cls(config=config)
        return guard.run_once_per_day(symbols)