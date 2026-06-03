#!/usr/bin/env python3
"""
Main CLI interface for the compliant financial news ingestion system.

Usage:
    python -m ingest.main enqueue --urls url1 url2 url3
    python -m ingest.main worker --workers 3
    python -m ingest.main status
"""

import asyncio
import argparse
import json
import sys
from typing import List

import structlog

from .scheduler import IngestionOrchestrator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def enqueue_urls(urls: List[str], config_path: str = "ingest/config.yaml"):
    """Enqueue URLs for processing"""
    orchestrator = IngestionOrchestrator(config_path)
    
    try:
        await orchestrator.initialize()
        
        job_ids = await orchestrator.enqueue_urls(urls)
        
        print(f"Enqueued {len(job_ids)} jobs:")
        for i, (url, job_id) in enumerate(zip(urls, job_ids)):
            print(f"  {i+1}. {url} -> {job_id}")
        
    finally:
        await orchestrator.shutdown()


async def run_workers(num_workers: int, config_path: str = "ingest/config.yaml"):
    """Run worker processes"""
    orchestrator = IngestionOrchestrator(config_path)
    
    try:
        await orchestrator.initialize()
        
        print(f"Starting {num_workers} workers...")
        await orchestrator.start_workers(num_workers)
        
        print("Workers started. Press Ctrl+C to stop.")
        
        # Print metrics every 30 seconds
        try:
            while True:
                await asyncio.sleep(30)
                metrics = await orchestrator.get_metrics()
                print(f"Metrics: {json.dumps(metrics, indent=2)}")
                
        except KeyboardInterrupt:
            print("\nShutting down workers...")
            await orchestrator.stop_workers()
    
    finally:
        await orchestrator.shutdown()


async def show_status(config_path: str = "ingest/config.yaml"):
    """Show system status and metrics"""
    orchestrator = IngestionOrchestrator(config_path)
    
    try:
        await orchestrator.initialize()
        
        metrics = await orchestrator.get_metrics()
        
        print("=== Ingestion System Status ===")
        print(f"Jobs processed: {metrics['jobs_processed']}")
        print(f"Jobs successful: {metrics['jobs_successful']}")
        print(f"Jobs failed: {metrics['jobs_failed']}")
        print(f"Articles extracted: {metrics['articles_extracted']}")
        print(f"Articles kept: {metrics['articles_kept']}")
        print(f"Articles rejected: {metrics['articles_rejected']}")
        
        print("\nQueue Status:")
        for queue_type, count in metrics['queues'].items():
            print(f"  {queue_type}: {count}")
        
        print(f"\nLast updated: {metrics['timestamp']}")
        
    finally:
        await orchestrator.shutdown()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Compliant Financial News Ingestion System"
    )
    parser.add_argument(
        '--config', 
        default='ingest/config.yaml',
        help='Path to configuration file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Enqueue command
    enqueue_parser = subparsers.add_parser('enqueue', help='Enqueue URLs for processing')
    enqueue_parser.add_argument('--urls', nargs='+', required=True, help='URLs to enqueue')
    enqueue_parser.add_argument('--priority', type=int, default=1, help='Job priority')
    
    # Worker command  
    worker_parser = subparsers.add_parser('worker', help='Run worker processes')
    worker_parser.add_argument('--workers', type=int, default=3, help='Number of workers')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo with sample financial news URLs')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'enqueue':
            asyncio.run(enqueue_urls(args.urls, args.config))
            
        elif args.command == 'worker':
            asyncio.run(run_workers(args.workers, args.config))
            
        elif args.command == 'status':
            asyncio.run(show_status(args.config))
            
        elif args.command == 'demo':
            # Sample financial news URLs for demonstration
            demo_urls = [
                'https://www.reuters.com/markets/',
                'https://www.cnbc.com/markets/',
                'https://finance.yahoo.com/news/',
                'https://www.marketwatch.com/latest-news',
            ]
            print("Running demo with sample financial news URLs...")
            print("Note: These are landing pages, not specific articles.")
            print("In a real implementation, you would provide specific article URLs.")
            print("\nDemo URLs:")
            for url in demo_urls:
                print(f"  - {url}")
            
            # Just enqueue for demo, don't actually process
            asyncio.run(enqueue_urls(demo_urls, args.config))
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error("Command failed", command=args.command, error=str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()