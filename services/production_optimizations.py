"""
Production Optimizations and Horizontal Scaling Support

Implements pandas DataFrame optimization, async I/O, connection pooling,
and horizontal scaling support for high-volume financial data scenarios.
"""

import asyncio
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
import aiohttp
import asyncpg
import aioredis
import uvloop
import pickle
import lz4.frame
import pyarrow as pa
import pyarrow.parquet as pq
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ScalingStrategy(Enum):
    VERTICAL = "vertical"      # Scale up single instance
    HORIZONTAL = "horizontal"  # Scale across multiple instances
    HYBRID = "hybrid"         # Combination of both

class OptimizationLevel(Enum):
    DEVELOPMENT = "development"  # Basic optimizations
    PRODUCTION = "production"    # Full production optimizations
    HIGH_VOLUME = "high_volume" # Maximum performance optimizations

@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    cpu_usage_percent: float = 0.0
    memory_usage_gb: float = 0.0
    disk_io_mb_per_sec: float = 0.0
    network_io_mb_per_sec: float = 0.0
    active_connections: int = 0
    avg_response_time_ms: float = 0.0
    throughput_requests_per_sec: float = 0.0
    cache_hit_rate: float = 0.0
    
@dataclass
class ScalingConfiguration:
    """Scaling and optimization configuration"""
    max_workers: int = multiprocessing.cpu_count()
    thread_pool_size: int = 50
    process_pool_size: int = 4
    connection_pool_size: int = 20
    async_concurrency_limit: int = 100
    dataframe_chunk_size: int = 10000
    parquet_compression: str = "snappy"
    redis_pipeline_size: int = 1000

class ProductionOptimizer:
    """Production-ready optimization and scaling manager"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.PRODUCTION):
        self.optimization_level = optimization_level
        self.config = self._get_optimization_config()
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Resource pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.connection_pools: Dict[str, Any] = {}
        
        # Async optimization
        self.session: Optional[aiohttp.ClientSession] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
        # DataFrame optimization settings
        self.parquet_engine = 'pyarrow'
        self.memory_map = True
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None

    def _get_optimization_config(self) -> ScalingConfiguration:
        """Get optimization configuration based on level"""
        
        if self.optimization_level == OptimizationLevel.DEVELOPMENT:
            return ScalingConfiguration(
                max_workers=2,
                thread_pool_size=10,
                process_pool_size=2,
                connection_pool_size=5,
                async_concurrency_limit=20,
                dataframe_chunk_size=5000
            )
        elif self.optimization_level == OptimizationLevel.HIGH_VOLUME:
            return ScalingConfiguration(
                max_workers=min(32, multiprocessing.cpu_count() * 2),
                thread_pool_size=200,
                process_pool_size=multiprocessor.cpu_count(),
                connection_pool_size=50,
                async_concurrency_limit=500,
                dataframe_chunk_size=50000,
                redis_pipeline_size=5000
            )
        else:  # PRODUCTION
            return ScalingConfiguration()

    async def initialize(self):
        """Initialize production optimizations"""
        
        logger.info(f"Initializing production optimizations (level: {self.optimization_level.value})")
        
        # Set event loop policy for better performance
        if os.name != 'nt':  # Not Windows
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        # Initialize resource pools
        await self._initialize_resource_pools()
        
        # Initialize async clients with optimization
        await self._initialize_async_clients()
        
        # Configure pandas for performance
        self._configure_pandas_optimization()
        
        # Start monitoring
        await self._start_monitoring()
        
        logger.info("Production optimizations initialized successfully")

    async def _initialize_resource_pools(self):
        """Initialize optimized resource pools"""
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size,
            thread_name_prefix="opt_thread"
        )
        
        # Process pool for heavy computations
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config.process_pool_size
        )
        
        logger.info(f"Initialized thread pool ({self.config.thread_pool_size} workers) "
                   f"and process pool ({self.config.process_pool_size} workers)")

    async def _initialize_async_clients(self):
        """Initialize optimized async clients"""
        
        # Optimized HTTP session
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        connector = aiohttp.TCPConnector(
            limit=self.config.connection_pool_size,
            limit_per_host=20,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'OptimizedFinancialSystem/1.0'}
        )
        
        # Redis client with optimizations
        try:
            self.redis_client = aioredis.from_url(
                "redis://localhost:6379",
                max_connections=self.config.connection_pool_size,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 3,  # TCP_KEEPINTVL  
                    3: 5   # TCP_KEEPCNT
                }
            )
            await self.redis_client.ping()
            logger.info("Redis client initialized with optimizations")
        except Exception as e:
            logger.warning(f"Redis optimization unavailable: {e}")
            self.redis_client = None

    def _configure_pandas_optimization(self):
        """Configure pandas for optimal performance"""
        
        # Memory optimizations
        pd.set_option('mode.copy_on_write', True)
        
        # Compute optimizations
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
        
        # Display optimizations
        pd.set_option('display.max_info_columns', 100)
        
        logger.info("Pandas configured with performance optimizations")

    async def _start_monitoring(self):
        """Start performance monitoring"""
        
        self._monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        self._optimization_task = asyncio.create_task(self._dynamic_optimization_loop())

    async def _performance_monitoring_loop(self):
        """Background performance monitoring"""
        
        while True:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def _collect_performance_metrics(self):
        """Collect comprehensive performance metrics"""
        
        try:
            # System metrics
            self.metrics.cpu_usage_percent = psutil.cpu_percent(interval=1)
            
            memory_info = psutil.virtual_memory()
            self.metrics.memory_usage_gb = (memory_info.total - memory_info.available) / 1024**3
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if hasattr(self, '_last_disk_io'):
                read_diff = disk_io.read_bytes - self._last_disk_io.read_bytes
                write_diff = disk_io.write_bytes - self._last_disk_io.write_bytes
                self.metrics.disk_io_mb_per_sec = (read_diff + write_diff) / 1024**2 / 30
            self._last_disk_io = disk_io
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                sent_diff = net_io.bytes_sent - self._last_net_io.bytes_sent
                recv_diff = net_io.bytes_recv - self._last_net_io.bytes_recv
                self.metrics.network_io_mb_per_sec = (sent_diff + recv_diff) / 1024**2 / 30
            self._last_net_io = net_io
            
            # Connection count (approximate)
            try:
                connections = len(psutil.net_connections())
                self.metrics.active_connections = connections
            except:
                pass
            
            # Store metrics history (keep last 100 points)
            self.metrics_history.append(self.metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")

    async def _dynamic_optimization_loop(self):
        """Dynamic optimization based on performance metrics"""
        
        while True:
            try:
                await self._adjust_optimizations()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dynamic optimization error: {e}")
                await asyncio.sleep(300)

    async def _adjust_optimizations(self):
        """Dynamically adjust optimizations based on current load"""
        
        if len(self.metrics_history) < 10:
            return
        
        # Calculate average metrics over last 10 readings
        recent_metrics = self.metrics_history[-10:]
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_gb for m in recent_metrics) / len(recent_metrics)
        
        # High CPU usage - reduce concurrent operations
        if avg_cpu > 80:
            new_limit = max(20, self.config.async_concurrency_limit // 2)
            if new_limit != self.config.async_concurrency_limit:
                self.config.async_concurrency_limit = new_limit
                logger.info(f"Reduced concurrency limit to {new_limit} due to high CPU usage")
        
        # High memory usage - trigger cleanup
        elif avg_memory > 8.0:  # 8GB threshold
            await self._memory_cleanup()
        
        # Low resource usage - increase limits
        elif avg_cpu < 40 and avg_memory < 4.0:
            new_limit = min(500, self.config.async_concurrency_limit + 50)
            if new_limit != self.config.async_concurrency_limit:
                self.config.async_concurrency_limit = new_limit
                logger.info(f"Increased concurrency limit to {new_limit}")

    async def _memory_cleanup(self):
        """Perform memory cleanup operations"""
        
        try:
            # Clear pandas caches
            pd._config.reset_option('^display\.')
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            
            logger.info(f"Memory cleanup completed, collected {collected} objects")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    # DataFrame Optimization Methods
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df[col] = df[col].astype('int32')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        # Optimize string columns
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        reduction_ratio = (original_memory - optimized_memory) / original_memory
        
        logger.info(f"DataFrame memory optimized: {reduction_ratio:.2%} reduction")
        
        return df

    async def save_dataframe_optimized(self, df: pd.DataFrame, file_path: str,
                                     compression: str = None) -> bool:
        """Save DataFrame with optimal format and compression"""
        
        try:
            file_path = Path(file_path)
            compression = compression or self.config.parquet_compression
            
            if file_path.suffix.lower() == '.parquet':
                # Use Parquet for optimal performance
                df.to_parquet(
                    file_path,
                    engine=self.parquet_engine,
                    compression=compression,
                    index=False
                )
            elif file_path.suffix.lower() == '.pkl':
                # Use optimized pickle with compression
                await self._save_pickle_compressed(df, file_path)
            else:
                # Fallback to CSV with compression
                df.to_csv(file_path, index=False, compression='gzip')
            
            logger.info(f"DataFrame saved to {file_path} with {compression} compression")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame: {e}")
            return False

    async def _save_pickle_compressed(self, df: pd.DataFrame, file_path: Path):
        """Save DataFrame as compressed pickle"""
        
        def _compress_and_save():
            pickled_data = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_data = lz4.frame.compress(pickled_data)
            
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
        
        await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, _compress_and_save
        )

    async def load_dataframe_optimized(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load DataFrame with optimal performance"""
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"File {file_path} does not exist")
                return None
            
            if file_path.suffix.lower() == '.parquet':
                # Load Parquet
                df = pd.read_parquet(file_path, engine=self.parquet_engine)
            elif file_path.suffix.lower() == '.pkl':
                # Load compressed pickle
                df = await self._load_pickle_compressed(file_path)
            else:
                # Load CSV with optimization
                df = pd.read_csv(file_path, low_memory=False)
                df = self.optimize_dataframe_memory(df)
            
            logger.info(f"DataFrame loaded from {file_path} ({len(df)} rows)")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load DataFrame from {file_path}: {e}")
            return None

    async def _load_pickle_compressed(self, file_path: Path) -> pd.DataFrame:
        """Load compressed pickle DataFrame"""
        
        def _load_and_decompress():
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            
            pickled_data = lz4.frame.decompress(compressed_data)
            return pickle.loads(pickled_data)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, _load_and_decompress
        )

    # Async I/O Optimization Methods
    
    async def batch_async_requests(self, urls: List[str], 
                                 headers: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Perform batch async HTTP requests with optimization"""
        
        semaphore = asyncio.Semaphore(self.config.async_concurrency_limit)
        
        async def fetch_url(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    async with self.session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {'url': url, 'success': True, 'data': data}
                        else:
                            return {'url': url, 'success': False, 'error': f'HTTP {response.status}'}
                except Exception as e:
                    return {'url': url, 'success': False, 'error': str(e)}
        
        # Execute requests with controlled concurrency
        tasks = [fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        success_count = sum(1 for r in valid_results if r.get('success', False))
        logger.info(f"Batch requests completed: {success_count}/{len(urls)} successful")
        
        return valid_results

    @asynccontextmanager
    async def database_connection_pool(self, dsn: str):
        """Optimized database connection pool context manager"""
        
        pool = await asyncpg.create_pool(
            dsn,
            min_size=5,
            max_size=self.config.connection_pool_size,
            command_timeout=60,
            server_settings={
                'application_name': 'optimized_financial_system',
                'work_mem': '256MB'
            }
        )
        
        try:
            yield pool
        finally:
            await pool.close()

    async def batch_database_operations(self, pool: asyncpg.Pool, 
                                      operations: List[Tuple[str, List]]) -> List[Any]:
        """Execute batch database operations efficiently"""
        
        async def execute_operation(conn: asyncpg.Connection, sql: str, params: List) -> Any:
            try:
                if sql.strip().upper().startswith('SELECT'):
                    return await conn.fetch(sql, *params)
                else:
                    return await conn.execute(sql, *params)
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                return None
        
        results = []
        async with pool.acquire() as conn:
            # Execute operations in batches
            for i in range(0, len(operations), 10):  # Process 10 at a time
                batch = operations[i:i+10]
                batch_tasks = [
                    execute_operation(conn, sql, params) 
                    for sql, params in batch
                ]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                results.extend(batch_results)
        
        return results

    # Redis Optimization Methods
    
    async def redis_pipeline_operations(self, operations: List[Tuple[str, List]]) -> List[Any]:
        """Execute Redis operations using optimized pipeline"""
        
        if not self.redis_client:
            logger.warning("Redis client not available")
            return []
        
        pipeline = self.redis_client.pipeline()
        
        try:
            # Add operations to pipeline
            for operation, args in operations:
                getattr(pipeline, operation)(*args)
            
            # Execute pipeline
            results = await pipeline.execute()
            
            logger.info(f"Redis pipeline executed {len(operations)} operations")
            return results
            
        except Exception as e:
            logger.error(f"Redis pipeline failed: {e}")
            return []

    # Horizontal Scaling Support
    
    async def distribute_workload(self, tasks: List[Any], 
                                worker_nodes: List[str]) -> List[Any]:
        """Distribute workload across multiple worker nodes"""
        
        if not worker_nodes:
            logger.warning("No worker nodes available for distribution")
            return []
        
        # Simple round-robin distribution
        distributed_tasks = []
        for i, task in enumerate(tasks):
            node = worker_nodes[i % len(worker_nodes)]
            distributed_tasks.append({
                'task': task,
                'worker_node': node,
                'task_id': f"task_{i}"
            })
        
        # Execute distributed tasks (placeholder implementation)
        results = []
        for dist_task in distributed_tasks:
            try:
                # In a real implementation, this would send tasks to worker nodes
                result = await self._execute_distributed_task(dist_task)
                results.append(result)
            except Exception as e:
                logger.error(f"Distributed task failed: {e}")
                results.append(None)
        
        return results

    async def _execute_distributed_task(self, dist_task: Dict[str, Any]) -> Any:
        """Execute a task on a distributed worker node"""
        
        # Placeholder implementation
        # In production, this would use message queues (Redis, RabbitMQ)
        # or direct HTTP calls to worker nodes
        
        worker_node = dist_task['worker_node']
        task = dist_task['task']
        
        logger.info(f"Executing task {dist_task['task_id']} on node {worker_node}")
        
        # Simulate task execution
        await asyncio.sleep(0.1)
        return f"Result from {worker_node}"

    # Performance Analysis Methods
    
    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze system performance and identify bottlenecks"""
        
        if len(self.metrics_history) < 10:
            return {'error': 'Insufficient performance data'}
        
        recent_metrics = self.metrics_history[-20:]
        
        analysis = {
            'cpu_analysis': {
                'avg_usage': sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics),
                'peak_usage': max(m.cpu_usage_percent for m in recent_metrics),
                'bottleneck': 'cpu' if sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics) > 70 else None
            },
            'memory_analysis': {
                'avg_usage_gb': sum(m.memory_usage_gb for m in recent_metrics) / len(recent_metrics),
                'peak_usage_gb': max(m.memory_usage_gb for m in recent_metrics),
                'bottleneck': 'memory' if sum(m.memory_usage_gb for m in recent_metrics) / len(recent_metrics) > 6 else None
            },
            'io_analysis': {
                'avg_disk_io': sum(m.disk_io_mb_per_sec for m in recent_metrics) / len(recent_metrics),
                'avg_network_io': sum(m.network_io_mb_per_sec for m in recent_metrics) / len(recent_metrics),
                'bottleneck': 'io' if sum(m.disk_io_mb_per_sec for m in recent_metrics) / len(recent_metrics) > 100 else None
            },
            'recommendations': self._generate_optimization_recommendations(recent_metrics)
        }
        
        return analysis

    def _generate_optimization_recommendations(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        
        recommendations = []
        
        avg_cpu = sum(m.cpu_usage_percent for m in metrics) / len(metrics)
        avg_memory = sum(m.memory_usage_gb for m in metrics) / len(metrics)
        avg_disk_io = sum(m.disk_io_mb_per_sec for m in metrics) / len(metrics)
        
        if avg_cpu > 80:
            recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
        
        if avg_memory > 8:
            recommendations.append("Implement more aggressive memory optimization and caching strategies")
        
        if avg_disk_io > 100:
            recommendations.append("Consider using faster storage (SSD) or implementing better data compression")
        
        if self.metrics.cache_hit_rate < 70:
            recommendations.append("Optimize caching strategy to improve hit rates")
        
        if not recommendations:
            recommendations.append("System performance is within optimal ranges")
        
        return recommendations

    async def get_optimization_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive optimization dashboard"""
        
        return {
            'current_metrics': {
                'cpu_usage_percent': self.metrics.cpu_usage_percent,
                'memory_usage_gb': round(self.metrics.memory_usage_gb, 2),
                'disk_io_mb_per_sec': round(self.metrics.disk_io_mb_per_sec, 2),
                'network_io_mb_per_sec': round(self.metrics.network_io_mb_per_sec, 2),
                'active_connections': self.metrics.active_connections,
                'avg_response_time_ms': round(self.metrics.avg_response_time_ms, 2)
            },
            'configuration': {
                'optimization_level': self.optimization_level.value,
                'max_workers': self.config.max_workers,
                'thread_pool_size': self.config.thread_pool_size,
                'connection_pool_size': self.config.connection_pool_size,
                'async_concurrency_limit': self.config.async_concurrency_limit
            },
            'performance_analysis': self.analyze_performance_bottlenecks()
        }

    async def cleanup(self):
        """Clean up resources"""
        
        logger.info("Cleaning up production optimizations")
        
        # Cancel monitoring tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        # Close async clients
        if self.session:
            await self.session.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        # Shutdown executor pools
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("Production optimizations cleanup completed")

# Factory function
async def create_production_optimizer(optimization_level: OptimizationLevel = OptimizationLevel.PRODUCTION) -> ProductionOptimizer:
    """Create and initialize production optimizer"""
    
    optimizer = ProductionOptimizer(optimization_level)
    await optimizer.initialize()
    
    logger.info(f"Production optimizer created with {optimization_level.value} level optimizations")
    return optimizer

# Add missing import
multiprocessor = multiprocessing