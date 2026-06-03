#!/usr/bin/env python3
"""
Simple file-based job queue for symbol intake processing
Manages job lifecycle and status tracking
"""
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from services.audit_logger import audit_logger

logger = logging.getLogger(__name__)

class JobQueue:
    """File-based job queue with status tracking"""
    
    def __init__(self):
        self.queue_dir = Path("data/job_queue")
        self.queue_dir.mkdir(exist_ok=True)
        
        # Job status directories
        self.queued_dir = self.queue_dir / "queued"
        self.processing_dir = self.queue_dir / "processing" 
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"
        
        for dir_path in [self.queued_dir, self.processing_dir, 
                        self.completed_dir, self.failed_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def enqueue_job(self, 
                    job_id: str,
                    job_type: str, 
                    job_data: Dict[str, Any],
                    priority: str = 'normal') -> bool:
        """Add job to queue"""
        with audit_logger.operation_context("job_queue", "enqueue_job", 
                                           symbol=job_data.get('symbol'),
                                           metadata={'job_type': job_type, 'priority': priority}) as op:
            try:
                job_record = {
                    'job_id': job_id,
                    'job_type': job_type,
                    'job_data': job_data,
                    'priority': priority,
                    'status': 'queued',
                    'created_at': time.time(),
                    'updated_at': time.time()
                }
                
                job_file = self.queued_dir / f"{job_id}.json"
                op.step("write_job_file", count_out=1, source_links=[str(job_file)])
                
                with open(job_file, 'w') as f:
                    json.dump(job_record, f, indent=2)
                
                op.step("job_enqueued", count_out=1, metadata={'job_id': job_id})
                logger.info(f"Enqueued job {job_id} ({job_type})")
                return True
                
            except Exception as e:
                logger.exception(f"Error enqueuing job {job_id}")
                return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a job"""
        # Check all status directories
        for status_dir, status in [
            (self.queued_dir, 'queued'),
            (self.processing_dir, 'processing'),
            (self.completed_dir, 'completed'),
            (self.failed_dir, 'failed')
        ]:
            job_file = status_dir / f"{job_id}.json"
            if job_file.exists():
                try:
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                    job_data['status'] = status
                    return job_data
                except Exception as e:
                    logger.error(f"Error reading job file {job_file}: {e}")
        
        return None
    
    def get_queue_size(self) -> int:
        """Get number of jobs in queue (queued + processing)"""
        try:
            queued_count = len(list(self.queued_dir.glob("*.json")))
            processing_count = len(list(self.processing_dir.glob("*.json")))
            return queued_count + processing_count
        except Exception as e:
            logger.error(f"Error getting queue size: {e}")
            return 0
    
    def move_job_to_processing(self, job_id: str) -> bool:
        """Move job from queued to processing"""
        return self._move_job(job_id, self.queued_dir, self.processing_dir, 'processing')
    
    def complete_job(self, job_id: str, result: Dict[str, Any] = None) -> bool:
        """Mark job as completed"""
        success = self._move_job(job_id, self.processing_dir, self.completed_dir, 'completed')
        if success and result:
            # Store job result
            self._update_job_data(job_id, self.completed_dir, {'result': result})
        return success
    
    def fail_job(self, job_id: str, error: str) -> bool:
        """Mark job as failed"""
        success = self._move_job(job_id, self.processing_dir, self.failed_dir, 'failed')
        if success:
            self._update_job_data(job_id, self.failed_dir, {'error': error})
        return success
    
    def _move_job(self, job_id: str, from_dir: Path, to_dir: Path, new_status: str) -> bool:
        """Move job between status directories"""
        try:
            from_file = from_dir / f"{job_id}.json"
            to_file = to_dir / f"{job_id}.json"
            
            if not from_file.exists():
                logger.warning(f"Job file {from_file} not found")
                return False
            
            # Read, update, and move
            with open(from_file, 'r') as f:
                job_data = json.load(f)
            
            job_data['status'] = new_status
            job_data['updated_at'] = time.time()
            
            with open(to_file, 'w') as f:
                json.dump(job_data, f, indent=2)
            
            from_file.unlink()  # Remove original
            return True
            
        except Exception as e:
            logger.exception(f"Error moving job {job_id} from {from_dir} to {to_dir}")
            return False
    
    def _update_job_data(self, job_id: str, job_dir: Path, updates: Dict[str, Any]) -> bool:
        """Update job data with additional information"""
        try:
            job_file = job_dir / f"{job_id}.json"
            if not job_file.exists():
                return False
            
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            
            job_data.update(updates)
            job_data['updated_at'] = time.time()
            
            with open(job_file, 'w') as f:
                json.dump(job_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.exception(f"Error updating job data for {job_id}")
            return False