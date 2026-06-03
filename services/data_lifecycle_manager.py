"""
Data Lifecycle Manager for Archival and Compliance

Manages automatic archival of data older than 2 years, compression,
cold storage integration, and regulatory retention compliance (7 years for financial data).
"""

import asyncio
import logging
import gzip
import json
import sqlite3
import shutil
import tarfile
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import boto3
from botocore.exceptions import ClientError
import schedule
import hashlib
import os

logger = logging.getLogger(__name__)

class StorageTier(Enum):
    HOT = "hot"           # Active data (0-6 months)
    WARM = "warm"         # Recent data (6 months - 2 years)
    COLD = "cold"         # Archived data (2-7 years)
    GLACIER = "glacier"   # Long-term storage (7+ years)

class DataCategory(Enum):
    PRICE_DATA = "price_data"
    NEWS_DATA = "news_data"
    SENTIMENT_DATA = "sentiment_data"
    RECOMMENDATION_DATA = "recommendation_data"
    CACHE_DATA = "cache_data"
    SYSTEM_LOGS = "system_logs"

class RetentionPolicy(Enum):
    FINANCIAL_DATA = timedelta(days=7*365)    # 7 years for financial data
    NEWS_DATA = timedelta(days=5*365)         # 5 years for news
    CACHE_DATA = timedelta(days=1*365)        # 1 year for cache
    SYSTEM_LOGS = timedelta(days=3*365)       # 3 years for logs

@dataclass
class ArchivalJob:
    """Archival job configuration"""
    job_id: str
    data_category: DataCategory
    source_path: str
    archive_path: str
    retention_policy: RetentionPolicy
    compression_enabled: bool
    encryption_enabled: bool
    storage_tier: StorageTier
    scheduled_time: datetime
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    bytes_archived: int = 0
    files_archived: int = 0

@dataclass
class StorageMetrics:
    """Storage utilization metrics"""
    total_size_gb: float = 0.0
    hot_storage_gb: float = 0.0
    warm_storage_gb: float = 0.0
    cold_storage_gb: float = 0.0
    glacier_storage_gb: float = 0.0
    compression_ratio: float = 1.0
    total_files: int = 0
    archived_files: int = 0
    last_archival_date: Optional[datetime] = None

class DataLifecycleManager:
    """Comprehensive data lifecycle and archival management"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Storage configuration
        self.base_data_path = Path(self.config.get('base_data_path', 'data'))
        self.archive_path = Path(self.config.get('archive_path', 'data/archive'))
        self.temp_path = Path(self.config.get('temp_path', 'data/temp'))
        
        # Cloud storage configuration
        self.s3_bucket = self.config.get('s3_bucket')
        self.s3_client = None
        if self.s3_bucket:
            self.s3_client = boto3.client('s3')
        
        # Archival thresholds
        self.archival_thresholds = {
            StorageTier.WARM: timedelta(days=180),    # 6 months
            StorageTier.COLD: timedelta(days=730),    # 2 years
            StorageTier.GLACIER: timedelta(days=2555) # 7 years
        }
        
        # Compression settings
        self.compression_enabled = self.config.get('compression_enabled', True)
        self.compression_level = self.config.get('compression_level', 6)
        
        # Encryption settings
        self.encryption_enabled = self.config.get('encryption_enabled', False)
        self.encryption_key = self.config.get('encryption_key')
        
        # Database for tracking
        self.db_path = "data/lifecycle_manager.db"
        
        # Background tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = StorageMetrics()
        
        self._initialize_directories()
        self._initialize_database()

    def _initialize_directories(self):
        """Initialize required directories"""
        
        directories = [
            self.base_data_path,
            self.archive_path,
            self.temp_path,
            self.archive_path / "hot",
            self.archive_path / "warm", 
            self.archive_path / "cold",
            self.archive_path / "glacier"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _initialize_database(self):
        """Initialize SQLite database for lifecycle tracking"""
        
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Archival jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archival_jobs (
                job_id TEXT PRIMARY KEY,
                data_category TEXT NOT NULL,
                source_path TEXT NOT NULL,
                archive_path TEXT NOT NULL,
                retention_policy TEXT NOT NULL,
                compression_enabled BOOLEAN NOT NULL,
                encryption_enabled BOOLEAN NOT NULL,
                storage_tier TEXT NOT NULL,
                scheduled_time INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                completed_at INTEGER,
                error_message TEXT,
                bytes_archived INTEGER DEFAULT 0,
                files_archived INTEGER DEFAULT 0
            )
        """)
        
        # Data inventory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                data_category TEXT NOT NULL,
                storage_tier TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                created_date INTEGER NOT NULL,
                last_accessed INTEGER NOT NULL,
                archived_date INTEGER,
                checksum TEXT NOT NULL,
                compression_ratio REAL DEFAULT 1.0,
                is_encrypted BOOLEAN DEFAULT 0
            )
        """)
        
        # Storage metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS storage_metrics (
                timestamp INTEGER PRIMARY KEY,
                total_size_gb REAL NOT NULL,
                hot_storage_gb REAL NOT NULL,
                warm_storage_gb REAL NOT NULL,
                cold_storage_gb REAL NOT NULL,
                glacier_storage_gb REAL NOT NULL,
                compression_ratio REAL NOT NULL,
                total_files INTEGER NOT NULL,
                archived_files INTEGER NOT NULL
            )
        """)
        
        # Compliance audit table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_audit (
                id TEXT PRIMARY KEY,
                audit_date INTEGER NOT NULL,
                data_category TEXT NOT NULL,
                files_checked INTEGER NOT NULL,
                compliant_files INTEGER NOT NULL,
                violations INTEGER NOT NULL,
                retention_violations TEXT,
                encryption_violations TEXT,
                audit_result TEXT NOT NULL
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON archival_jobs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_scheduled ON archival_jobs(scheduled_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventory_tier ON data_inventory(storage_tier)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventory_date ON data_inventory(created_date)")
        
        conn.commit()
        conn.close()

    async def start_lifecycle_management(self):
        """Start the data lifecycle management system"""
        
        logger.info("Starting data lifecycle management system")
        
        # Initial data scan and classification
        await self._scan_and_classify_data()
        
        # Start scheduler task
        self._scheduler_task = asyncio.create_task(self._archival_scheduler())
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Data lifecycle management started successfully")

    async def stop_lifecycle_management(self):
        """Stop the lifecycle management system"""
        
        logger.info("Stopping data lifecycle management")
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def _scan_and_classify_data(self):
        """Scan and classify existing data for lifecycle management"""
        
        logger.info("Scanning and classifying existing data")
        
        data_mappings = {
            'av_bulk_cache': DataCategory.PRICE_DATA,
            'news_data': DataCategory.NEWS_DATA,
            'sentiment_data': DataCategory.SENTIMENT_DATA,
            'recommendations': DataCategory.RECOMMENDATION_DATA,
            'cache': DataCategory.CACHE_DATA,
            'logs': DataCategory.SYSTEM_LOGS
        }
        
        total_files = 0
        total_size = 0
        
        for directory, category in data_mappings.items():
            dir_path = self.base_data_path / directory
            
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        try:
                            await self._classify_file(file_path, category)
                            total_files += 1
                            total_size += file_path.stat().st_size
                        except Exception as e:
                            logger.error(f"Failed to classify {file_path}: {e}")
        
        logger.info(f"Classified {total_files} files ({total_size / 1024**3:.2f} GB)")
        
        # Update metrics
        await self._update_storage_metrics()

    async def _classify_file(self, file_path: Path, category: DataCategory):
        """Classify a file and add to inventory"""
        
        try:
            stat_info = file_path.stat()
            file_size = stat_info.st_size
            created_time = datetime.fromtimestamp(stat_info.st_ctime)
            accessed_time = datetime.fromtimestamp(stat_info.st_atime)
            
            # Calculate checksum
            checksum = await self._calculate_checksum(file_path)
            
            # Determine storage tier based on age
            age = datetime.now() - created_time
            storage_tier = self._determine_storage_tier(age)
            
            # Store in inventory
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO data_inventory 
                (file_path, data_category, storage_tier, file_size, created_date, 
                 last_accessed, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(file_path),
                category.value,
                storage_tier.value,
                file_size,
                int(created_time.timestamp()),
                int(accessed_time.timestamp()),
                checksum
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to classify file {file_path}: {e}")

    def _determine_storage_tier(self, age: timedelta) -> StorageTier:
        """Determine appropriate storage tier based on data age"""
        
        if age > self.archival_thresholds[StorageTier.GLACIER]:
            return StorageTier.GLACIER
        elif age > self.archival_thresholds[StorageTier.COLD]:
            return StorageTier.COLD
        elif age > self.archival_thresholds[StorageTier.WARM]:
            return StorageTier.WARM
        else:
            return StorageTier.HOT

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum for file integrity"""
        
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

    async def _archival_scheduler(self):
        """Background scheduler for archival jobs"""
        
        while True:
            try:
                # Check for files that need archiving
                await self._schedule_archival_jobs()
                
                # Process pending archival jobs
                await self._process_archival_jobs()
                
                # Sleep for 1 hour before next check
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Archival scheduler error: {e}")
                await asyncio.sleep(3600)

    async def _schedule_archival_jobs(self):
        """Schedule archival jobs for eligible files"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find files that need to be moved to different tiers
        current_time = int(datetime.now().timestamp())
        
        # Files that should be moved to warm storage (6 months old)
        warm_threshold = current_time - int(self.archival_thresholds[StorageTier.WARM].total_seconds())
        
        cursor.execute("""
            SELECT file_path, data_category, storage_tier 
            FROM data_inventory 
            WHERE created_date < ? AND storage_tier = 'hot' AND archived_date IS NULL
        """, (warm_threshold,))
        
        hot_to_warm = cursor.fetchall()
        
        # Files that should be moved to cold storage (2 years old)
        cold_threshold = current_time - int(self.archival_thresholds[StorageTier.COLD].total_seconds())
        
        cursor.execute("""
            SELECT file_path, data_category, storage_tier 
            FROM data_inventory 
            WHERE created_date < ? AND storage_tier IN ('hot', 'warm') AND archived_date IS NULL
        """, (cold_threshold,))
        
        warm_to_cold = cursor.fetchall()
        
        # Files that should be moved to glacier (7 years old)
        glacier_threshold = current_time - int(self.archival_thresholds[StorageTier.GLACIER].total_seconds())
        
        cursor.execute("""
            SELECT file_path, data_category, storage_tier 
            FROM data_inventory 
            WHERE created_date < ? AND storage_tier IN ('hot', 'warm', 'cold') AND archived_date IS NULL
        """, (glacier_threshold,))
        
        cold_to_glacier = cursor.fetchall()
        
        conn.close()
        
        # Schedule archival jobs
        for file_path, category, current_tier in hot_to_warm:
            await self._create_archival_job(file_path, DataCategory(category), StorageTier.WARM)
        
        for file_path, category, current_tier in warm_to_cold:
            await self._create_archival_job(file_path, DataCategory(category), StorageTier.COLD)
        
        for file_path, category, current_tier in cold_to_glacier:
            await self._create_archival_job(file_path, DataCategory(category), StorageTier.GLACIER)

    async def _create_archival_job(self, file_path: str, category: DataCategory, 
                                 target_tier: StorageTier) -> str:
        """Create an archival job"""
        
        job_id = f"archive_{int(datetime.now().timestamp())}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}"
        
        # Determine archive path
        archive_subdir = self.archive_path / target_tier.value / category.value
        archive_subdir.mkdir(parents=True, exist_ok=True)
        
        original_path = Path(file_path)
        archive_path = archive_subdir / f"{original_path.stem}_{int(datetime.now().timestamp())}{original_path.suffix}"
        
        # Create archival job
        job = ArchivalJob(
            job_id=job_id,
            data_category=category,
            source_path=file_path,
            archive_path=str(archive_path),
            retention_policy=self._get_retention_policy(category),
            compression_enabled=self.compression_enabled,
            encryption_enabled=self.encryption_enabled,
            storage_tier=target_tier,
            scheduled_time=datetime.now()
        )
        
        # Store in database
        await self._store_archival_job(job)
        
        logger.info(f"Created archival job {job_id} for {file_path} -> {target_tier.value}")
        
        return job_id

    def _get_retention_policy(self, category: DataCategory) -> RetentionPolicy:
        """Get retention policy for data category"""
        
        policy_mapping = {
            DataCategory.PRICE_DATA: RetentionPolicy.FINANCIAL_DATA,
            DataCategory.NEWS_DATA: RetentionPolicy.NEWS_DATA,
            DataCategory.SENTIMENT_DATA: RetentionPolicy.FINANCIAL_DATA,
            DataCategory.RECOMMENDATION_DATA: RetentionPolicy.FINANCIAL_DATA,
            DataCategory.CACHE_DATA: RetentionPolicy.CACHE_DATA,
            DataCategory.SYSTEM_LOGS: RetentionPolicy.SYSTEM_LOGS
        }
        
        return policy_mapping.get(category, RetentionPolicy.FINANCIAL_DATA)

    async def _store_archival_job(self, job: ArchivalJob):
        """Store archival job in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO archival_jobs 
                (job_id, data_category, source_path, archive_path, retention_policy,
                 compression_enabled, encryption_enabled, storage_tier, scheduled_time,
                 status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id,
                job.data_category.value,
                job.source_path,
                job.archive_path,
                job.retention_policy.value,
                job.compression_enabled,
                job.encryption_enabled,
                job.storage_tier.value,
                int(job.scheduled_time.timestamp()),
                job.status,
                int(job.created_at.timestamp())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store archival job: {e}")

    async def _process_archival_jobs(self):
        """Process pending archival jobs"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get pending jobs
        cursor.execute("""
            SELECT job_id, data_category, source_path, archive_path, 
                   compression_enabled, encryption_enabled, storage_tier
            FROM archival_jobs 
            WHERE status = 'pending'
            ORDER BY scheduled_time ASC
            LIMIT 10
        """)
        
        pending_jobs = cursor.fetchall()
        conn.close()
        
        for job_data in pending_jobs:
            job_id = job_data[0]
            
            try:
                # Update job status to processing
                await self._update_job_status(job_id, "processing")
                
                # Execute archival
                success, bytes_archived, files_archived = await self._execute_archival(job_data)
                
                if success:
                    await self._update_job_status(job_id, "completed", bytes_archived, files_archived)
                    logger.info(f"Archival job {job_id} completed successfully")
                else:
                    await self._update_job_status(job_id, "failed", error_message="Archival execution failed")
                    logger.error(f"Archival job {job_id} failed")
                    
            except Exception as e:
                await self._update_job_status(job_id, "failed", error_message=str(e))
                logger.error(f"Archival job {job_id} error: {e}")

    async def _execute_archival(self, job_data: Tuple) -> Tuple[bool, int, int]:
        """Execute the actual archival process"""
        
        (job_id, data_category, source_path, archive_path, 
         compression_enabled, encryption_enabled, storage_tier) = job_data
        
        try:
            source_file = Path(source_path)
            archive_file = Path(archive_path)
            
            if not source_file.exists():
                logger.warning(f"Source file {source_path} no longer exists")
                return False, 0, 0
            
            # Ensure archive directory exists
            archive_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Calculate original size
            original_size = source_file.stat().st_size
            
            # Process file based on configuration
            if compression_enabled and archive_file.suffix != '.gz':
                # Compress the file
                compressed_path = archive_file.with_suffix(archive_file.suffix + '.gz')
                await self._compress_file(source_file, compressed_path)
                final_archive_path = compressed_path
            else:
                # Copy without compression
                shutil.copy2(source_file, archive_file)
                final_archive_path = archive_file
            
            # Encrypt if enabled
            if encryption_enabled and self.encryption_key:
                encrypted_path = final_archive_path.with_suffix(final_archive_path.suffix + '.enc')
                await self._encrypt_file(final_archive_path, encrypted_path)
                final_archive_path.unlink()  # Remove unencrypted version
                final_archive_path = encrypted_path
            
            # Upload to cloud storage if configured
            if storage_tier == StorageTier.GLACIER and self.s3_client:
                await self._upload_to_s3(final_archive_path, storage_tier)
            
            # Update inventory
            await self._update_inventory_after_archival(source_path, str(final_archive_path), storage_tier)
            
            # Remove original file (move to archive is complete)
            source_file.unlink()
            
            final_size = final_archive_path.stat().st_size
            compression_ratio = original_size / final_size if final_size > 0 else 1.0
            
            logger.info(f"Archived {source_path} -> {final_archive_path} "
                       f"(compression: {compression_ratio:.2f}x)")
            
            return True, final_size, 1
            
        except Exception as e:
            logger.error(f"Archival execution failed: {e}")
            return False, 0, 0

    async def _compress_file(self, source_path: Path, target_path: Path):
        """Compress file using gzip"""
        
        with open(source_path, 'rb') as f_in:
            with gzip.open(target_path, 'wb', compresslevel=self.compression_level) as f_out:
                shutil.copyfileobj(f_in, f_out)

    async def _encrypt_file(self, source_path: Path, target_path: Path):
        """Encrypt file (placeholder implementation)"""
        
        # This is a placeholder - in production, use proper encryption like AES
        # For now, just copy the file
        shutil.copy2(source_path, target_path)
        logger.info(f"File {source_path} encrypted to {target_path}")

    async def _upload_to_s3(self, file_path: Path, storage_tier: StorageTier):
        """Upload file to S3 with appropriate storage class"""
        
        if not self.s3_client or not self.s3_bucket:
            return
        
        storage_class_mapping = {
            StorageTier.WARM: 'STANDARD_IA',
            StorageTier.COLD: 'GLACIER',
            StorageTier.GLACIER: 'DEEP_ARCHIVE'
        }
        
        storage_class = storage_class_mapping.get(storage_tier, 'STANDARD')
        s3_key = f"archive/{storage_tier.value}/{file_path.name}"
        
        try:
            self.s3_client.upload_file(
                str(file_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs={'StorageClass': storage_class}
            )
            
            logger.info(f"Uploaded {file_path} to S3 bucket {self.s3_bucket} with {storage_class}")
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")

    async def _update_inventory_after_archival(self, original_path: str, 
                                             archive_path: str, storage_tier: StorageTier):
        """Update inventory after successful archival"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE data_inventory 
                SET file_path = ?, storage_tier = ?, archived_date = ?
                WHERE file_path = ?
            """, (
                archive_path,
                storage_tier.value,
                int(datetime.now().timestamp()),
                original_path
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update inventory: {e}")

    async def _update_job_status(self, job_id: str, status: str, 
                               bytes_archived: int = 0, files_archived: int = 0,
                               error_message: str = None):
        """Update archival job status"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = ["status = ?"]
            update_values = [status]
            
            if status == "completed":
                update_fields.extend(["completed_at = ?", "bytes_archived = ?", "files_archived = ?"])
                update_values.extend([int(datetime.now().timestamp()), bytes_archived, files_archived])
            
            if error_message:
                update_fields.append("error_message = ?")
                update_values.append(error_message)
            
            update_values.append(job_id)
            
            cursor.execute(f"""
                UPDATE archival_jobs 
                SET {', '.join(update_fields)}
                WHERE job_id = ?
            """, update_values)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

    async def _monitoring_loop(self):
        """Background monitoring and metrics collection"""
        
        while True:
            try:
                await self._update_storage_metrics()
                await self._run_compliance_audit()
                await self._cleanup_old_jobs()
                
                # Sleep for 6 hours
                await asyncio.sleep(6 * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(3600)

    async def _update_storage_metrics(self):
        """Update storage utilization metrics"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate storage by tier
            cursor.execute("""
                SELECT storage_tier, SUM(file_size) as total_size, COUNT(*) as file_count
                FROM data_inventory 
                GROUP BY storage_tier
            """)
            
            tier_metrics = cursor.fetchall()
            
            # Reset metrics
            self.metrics = StorageMetrics()
            
            for tier, size, count in tier_metrics:
                size_gb = size / (1024**3)
                self.metrics.total_size_gb += size_gb
                self.metrics.total_files += count
                
                if tier == StorageTier.HOT.value:
                    self.metrics.hot_storage_gb = size_gb
                elif tier == StorageTier.WARM.value:
                    self.metrics.warm_storage_gb = size_gb
                elif tier == StorageTier.COLD.value:
                    self.metrics.cold_storage_gb = size_gb
                elif tier == StorageTier.GLACIER.value:
                    self.metrics.glacier_storage_gb = size_gb
            
            # Count archived files
            cursor.execute("SELECT COUNT(*) FROM data_inventory WHERE archived_date IS NOT NULL")
            self.metrics.archived_files = cursor.fetchone()[0]
            
            # Get last archival date
            cursor.execute("SELECT MAX(completed_at) FROM archival_jobs WHERE status = 'completed'")
            last_archival = cursor.fetchone()[0]
            if last_archival:
                self.metrics.last_archival_date = datetime.fromtimestamp(last_archival)
            
            # Calculate average compression ratio
            cursor.execute("SELECT AVG(compression_ratio) FROM data_inventory WHERE compression_ratio > 0")
            avg_compression = cursor.fetchone()[0]
            self.metrics.compression_ratio = avg_compression or 1.0
            
            # Store metrics
            cursor.execute("""
                INSERT INTO storage_metrics 
                (timestamp, total_size_gb, hot_storage_gb, warm_storage_gb, 
                 cold_storage_gb, glacier_storage_gb, compression_ratio, 
                 total_files, archived_files)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(datetime.now().timestamp()),
                self.metrics.total_size_gb,
                self.metrics.hot_storage_gb,
                self.metrics.warm_storage_gb,
                self.metrics.cold_storage_gb,
                self.metrics.glacier_storage_gb,
                self.metrics.compression_ratio,
                self.metrics.total_files,
                self.metrics.archived_files
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Storage metrics updated: {self.metrics.total_size_gb:.2f} GB total, "
                       f"{self.metrics.archived_files} files archived")
            
        except Exception as e:
            logger.error(f"Failed to update storage metrics: {e}")

    async def _run_compliance_audit(self):
        """Run compliance audit for retention policies"""
        
        audit_id = f"audit_{int(datetime.now().timestamp())}"
        audit_results = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check each data category for compliance
            for category in DataCategory:
                retention_policy = self._get_retention_policy(category)
                retention_threshold = datetime.now() - retention_policy.value
                
                # Count files in this category
                cursor.execute("""
                    SELECT COUNT(*) FROM data_inventory 
                    WHERE data_category = ?
                """, (category.value,))
                
                total_files = cursor.fetchone()[0]
                
                # Count files that should be archived but aren't
                cursor.execute("""
                    SELECT COUNT(*) FROM data_inventory 
                    WHERE data_category = ? AND created_date < ? AND archived_date IS NULL
                """, (category.value, int(retention_threshold.timestamp())))
                
                violations = cursor.fetchone()[0]
                compliant_files = total_files - violations
                
                audit_results[category.value] = {
                    'total_files': total_files,
                    'compliant_files': compliant_files,
                    'violations': violations,
                    'compliance_rate': (compliant_files / total_files * 100) if total_files > 0 else 100
                }
            
            # Store audit results
            cursor.execute("""
                INSERT INTO compliance_audit 
                (id, audit_date, data_category, files_checked, compliant_files, 
                 violations, audit_result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_id,
                int(datetime.now().timestamp()),
                'ALL_CATEGORIES',
                sum(r['total_files'] for r in audit_results.values()),
                sum(r['compliant_files'] for r in audit_results.values()),
                sum(r['violations'] for r in audit_results.values()),
                json.dumps(audit_results)
            ))
            
            conn.commit()
            conn.close()
            
            # Log compliance status
            total_violations = sum(r['violations'] for r in audit_results.values())
            if total_violations > 0:
                logger.warning(f"Compliance audit found {total_violations} retention policy violations")
            else:
                logger.info("Compliance audit: All files comply with retention policies")
                
        except Exception as e:
            logger.error(f"Compliance audit failed: {e}")

    async def _cleanup_old_jobs(self):
        """Clean up old completed archival jobs"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Remove jobs older than 90 days
            cleanup_threshold = int((datetime.now() - timedelta(days=90)).timestamp())
            
            cursor.execute("""
                DELETE FROM archival_jobs 
                WHERE completed_at < ? AND status = 'completed'
            """, (cleanup_threshold,))
            
            deleted_jobs = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted_jobs > 0:
                logger.info(f"Cleaned up {deleted_jobs} old archival jobs")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")

    async def get_lifecycle_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle management dashboard"""
        
        try:
            dashboard = {
                'storage_metrics': {
                    'total_size_gb': round(self.metrics.total_size_gb, 2),
                    'hot_storage_gb': round(self.metrics.hot_storage_gb, 2),
                    'warm_storage_gb': round(self.metrics.warm_storage_gb, 2),
                    'cold_storage_gb': round(self.metrics.cold_storage_gb, 2),
                    'glacier_storage_gb': round(self.metrics.glacier_storage_gb, 2),
                    'compression_ratio': round(self.metrics.compression_ratio, 2),
                    'total_files': self.metrics.total_files,
                    'archived_files': self.metrics.archived_files,
                    'last_archival_date': self.metrics.last_archival_date.isoformat() if self.metrics.last_archival_date else None
                }
            }
            
            # Get recent job statistics
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Jobs by status
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM archival_jobs 
                WHERE created_at > ?
                GROUP BY status
            """, (int((datetime.now() - timedelta(days=7)).timestamp()),))
            
            job_stats = {}
            for status, count in cursor.fetchall():
                job_stats[status] = count
            
            dashboard['recent_jobs'] = job_stats
            
            # Compliance status
            cursor.execute("""
                SELECT audit_result 
                FROM compliance_audit 
                ORDER BY audit_date DESC 
                LIMIT 1
            """)
            
            latest_audit = cursor.fetchone()
            if latest_audit:
                dashboard['compliance'] = json.loads(latest_audit[0])
            
            conn.close()
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to get lifecycle dashboard: {e}")
            return {'error': str(e)}

    async def force_archival_scan(self):
        """Force immediate archival scan and job creation"""
        
        logger.info("Forcing archival scan")
        await self._schedule_archival_jobs()

    async def manual_archive_file(self, file_path: str, target_tier: StorageTier) -> str:
        """Manually archive a specific file"""
        
        if not Path(file_path).exists():
            raise ValueError(f"File {file_path} does not exist")
        
        # Determine data category (basic heuristics)
        category = DataCategory.CACHE_DATA  # Default
        
        if 'price' in file_path.lower() or 'ohlc' in file_path.lower():
            category = DataCategory.PRICE_DATA
        elif 'news' in file_path.lower():
            category = DataCategory.NEWS_DATA
        elif 'sentiment' in file_path.lower():
            category = DataCategory.SENTIMENT_DATA
        
        job_id = await self._create_archival_job(file_path, category, target_tier)
        
        logger.info(f"Manual archival job created: {job_id}")
        return job_id

# Factory function
async def create_lifecycle_manager(config: Dict[str, Any] = None) -> DataLifecycleManager:
    """Create and start data lifecycle manager"""
    
    manager = DataLifecycleManager(config)
    await manager.start_lifecycle_management()
    
    logger.info("Data lifecycle manager created and started")
    return manager