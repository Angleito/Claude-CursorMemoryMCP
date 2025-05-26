#!/usr/bin/env python3
"""
Advanced Query Performance Tuning for pgvector and mem0ai
Production-grade query optimization and performance monitoring
"""

import asyncio
import asyncpg
import time
import logging
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import psutil
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries to optimize"""
    SIMILARITY_SEARCH = "similarity_search"
    BATCH_INSERT = "batch_insert"
    BATCH_UPDATE = "batch_update"
    INDEX_SCAN = "index_scan"
    AGGREGATION = "aggregation"
    JOIN_QUERY = "join_query"

@dataclass
class QueryPerformanceConfig:
    """Configuration for query performance tuning"""
    # Connection pool settings
    min_connections: int = 10
    max_connections: int = 50
    connection_timeout: int = 60
    
    # Query optimization settings
    work_mem_mb: int = 256
    maintenance_work_mem_mb: int = 512
    shared_buffers_mb: int = 1024
    effective_cache_size_mb: int = 4096
    random_page_cost: float = 1.1
    seq_page_cost: float = 1.0
    
    # Vector-specific settings
    hnsw_ef_search: int = 64
    ivf_probes: int = 10
    max_parallel_workers: int = 4
    max_parallel_workers_per_gather: int = 2
    
    # Monitoring settings
    enable_query_logging: bool = True
    slow_query_threshold_ms: int = 1000
    enable_auto_explain: bool = True
    auto_explain_min_duration: int = 500

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_type: str
    execution_time_ms: float
    planning_time_ms: float
    rows_returned: int
    index_used: Optional[str]
    buffer_hits: int
    buffer_misses: int
    cpu_usage_percent: float
    memory_usage_mb: float
    timestamp: datetime

@dataclass
class IndexAnalysis:
    """Index usage and performance analysis"""
    index_name: str
    table_name: str
    index_type: str
    size_mb: float
    usage_count: int
    last_used: Optional[datetime]
    efficiency_score: float
    recommendations: List[str]

class PostgreSQLOptimizer:
    """PostgreSQL and pgvector query optimizer"""
    
    def __init__(self, db_url: str, config: QueryPerformanceConfig = None):
        self.db_url = db_url
        self.config = config or QueryPerformanceConfig()
        self.pool = None
        self.baseline_metrics = {}
        
    async def initialize(self):
        """Initialize the optimizer and database connection"""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
            command_timeout=self.config.connection_timeout
        )
        
        # Apply initial optimizations
        await self._apply_postgresql_optimizations()
        await self._setup_monitoring()
        
        logger.info("PostgreSQL optimizer initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.pool:
            await self.pool.close()
    
    async def _apply_postgresql_optimizations(self):
        """Apply PostgreSQL configuration optimizations"""
        optimizations = [
            f"SET work_mem = '{self.config.work_mem_mb}MB'",
            f"SET maintenance_work_mem = '{self.config.maintenance_work_mem_mb}MB'",
            f"SET effective_cache_size = '{self.config.effective_cache_size_mb}MB'",
            f"SET random_page_cost = {self.config.random_page_cost}",
            f"SET seq_page_cost = {self.config.seq_page_cost}",
            f"SET max_parallel_workers_per_gather = {self.config.max_parallel_workers_per_gather}",
            "SET enable_seqscan = on",
            "SET enable_indexscan = on",
            "SET enable_bitmapscan = on",
            "SET enable_nestloop = on",
            "SET enable_hashjoin = on",
            "SET enable_mergejoin = on"
        ]
        
        async with self.pool.acquire() as conn:
            for optimization in optimizations:
                try:
                    await conn.execute(optimization)
                    logger.debug(f"Applied: {optimization}")
                except Exception as e:
                    logger.warning(f"Failed to apply {optimization}: {e}")
    
    async def _setup_monitoring(self):
        """Setup query monitoring and logging"""
        async with self.pool.acquire() as conn:
            try:
                # Enable pg_stat_statements if available
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
                
                # Reset statistics
                await conn.execute("SELECT pg_stat_statements_reset()")
                
                # Configure auto_explain if enabled
                if self.config.enable_auto_explain:
                    await conn.execute("LOAD 'auto_explain'")
                    await conn.execute(f"SET auto_explain.log_min_duration = {self.config.auto_explain_min_duration}")
                    await conn.execute("SET auto_explain.log_analyze = true")
                    await conn.execute("SET auto_explain.log_buffers = true")
                    await conn.execute("SET auto_explain.log_verbose = true")
                
                logger.info("Query monitoring enabled")
                
            except Exception as e:
                logger.warning(f"Failed to setup monitoring: {e}")
    
    async def analyze_query_performance(self, query: str, params: List = None) -> QueryMetrics:
        """Analyze performance of a specific query"""
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        async with self.pool.acquire() as conn:
            # Explain analyze the query
            explain_query = f"EXPLAIN (ANALYZE true, BUFFERS true, FORMAT json) {query}"
            
            try:
                if params:
                    explain_result = await conn.fetchval(explain_query, *params)
                else:
                    explain_result = await conn.fetchval(explain_query)
                
                execution_time = (time.time() - start_time) * 1000
                end_cpu = psutil.cpu_percent()
                end_memory = psutil.virtual_memory().used / (1024 * 1024)
                
                # Parse explain output
                plan = explain_result[0]
                execution_time_db = plan.get('Execution Time', 0)
                planning_time = plan.get('Planning Time', 0)
                
                # Extract buffer statistics
                buffer_info = self._extract_buffer_stats(plan)
                
                # Determine index usage
                index_used = self._extract_index_usage(plan)
                
                return QueryMetrics(
                    query_type=self._classify_query(query),
                    execution_time_ms=execution_time_db,
                    planning_time_ms=planning_time,
                    rows_returned=plan.get('Plan', {}).get('Actual Rows', 0),
                    index_used=index_used,
                    buffer_hits=buffer_info.get('hits', 0),
                    buffer_misses=buffer_info.get('misses', 0),
                    cpu_usage_percent=end_cpu - start_cpu,
                    memory_usage_mb=end_memory - start_memory,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                logger.error(f"Query analysis failed: {e}")
                raise
    
    def _extract_buffer_stats(self, plan: Dict) -> Dict[str, int]:
        """Extract buffer hit/miss statistics from explain plan"""
        stats = {'hits': 0, 'misses': 0}
        
        def traverse_plan(node):
            if isinstance(node, dict):
                stats['hits'] += node.get('Shared Hit Blocks', 0)
                stats['misses'] += node.get('Shared Read Blocks', 0)
                
                for key, value in node.items():
                    if key == 'Plans' and isinstance(value, list):
                        for subplan in value:
                            traverse_plan(subplan)
                    elif isinstance(value, dict):
                        traverse_plan(value)
        
        traverse_plan(plan)
        return stats
    
    def _extract_index_usage(self, plan: Dict) -> Optional[str]:
        """Extract index usage from explain plan"""
        def find_index(node):
            if isinstance(node, dict):
                node_type = node.get('Node Type', '')
                if 'Index' in node_type:
                    return node.get('Index Name')
                
                # Check child plans
                plans = node.get('Plans', [])
                for subplan in plans:
                    index_name = find_index(subplan)
                    if index_name:
                        return index_name
            return None
        
        return find_index(plan.get('Plan', {}))
    
    def _classify_query(self, query: str) -> str:
        """Classify query type based on SQL content"""
        query_lower = query.lower().strip()
        
        if 'select' in query_lower and ('<->' in query_lower or '<#>' in query_lower):
            return QueryType.SIMILARITY_SEARCH.value
        elif query_lower.startswith('insert'):
            return QueryType.BATCH_INSERT.value
        elif query_lower.startswith('update'):
            return QueryType.BATCH_UPDATE.value
        elif 'join' in query_lower:
            return QueryType.JOIN_QUERY.value
        elif any(agg in query_lower for agg in ['count', 'sum', 'avg', 'max', 'min']):
            return QueryType.AGGREGATION.value
        else:
            return QueryType.INDEX_SCAN.value
    
    async def optimize_similarity_search(self, user_id: str, embedding: List[float], 
                                       top_k: int = 10) -> Dict[str, Any]:
        """Optimize similarity search queries"""
        results = {}
        
        # Test different query variations
        queries = [
            # Basic cosine similarity
            ("cosine_basic", """
                SELECT id, memory_text, 1 - (embedding <=> $2) as similarity
                FROM mem0_vectors.memories 
                WHERE user_id = $1 
                ORDER BY embedding <=> $2 
                LIMIT $3
            """),
            
            # With similarity threshold
            ("cosine_threshold", """
                SELECT id, memory_text, 1 - (embedding <=> $2) as similarity
                FROM mem0_vectors.memories 
                WHERE user_id = $1 AND 1 - (embedding <=> $2) > 0.7
                ORDER BY embedding <=> $2 
                LIMIT $3
            """),
            
            # Using inner product
            ("inner_product", """
                SELECT id, memory_text, (embedding <#> $2) as similarity
                FROM mem0_vectors.memories 
                WHERE user_id = $1 
                ORDER BY embedding <#> $2 DESC
                LIMIT $3
            """),
            
            # With additional filters
            ("filtered_search", """
                SELECT id, memory_text, 1 - (embedding <=> $2) as similarity
                FROM mem0_vectors.memories 
                WHERE user_id = $1 
                    AND importance_score > 0.5
                    AND created_at > NOW() - INTERVAL '30 days'
                ORDER BY embedding <=> $2 
                LIMIT $3
            """)
        ]
        
        for query_name, query_sql in queries:
            try:
                metrics = await self.analyze_query_performance(
                    query_sql, [user_id, embedding, top_k]
                )
                results[query_name] = asdict(metrics)
                
            except Exception as e:
                logger.error(f"Failed to analyze {query_name}: {e}")
                results[query_name] = {'error': str(e)}
        
        return results
    
    async def analyze_index_performance(self) -> List[IndexAnalysis]:
        """Analyze index usage and performance"""
        async with self.pool.acquire() as conn:
            # Get index statistics
            index_stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan as usage_count,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes 
                WHERE schemaname = 'mem0_vectors'
                ORDER BY idx_scan DESC
            """)
            
            # Get index sizes
            index_sizes = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    pg_size_pretty(pg_relation_size(indexrelid)) as size_pretty,
                    pg_relation_size(indexrelid) as size_bytes
                FROM pg_stat_user_indexes 
                WHERE schemaname = 'mem0_vectors'
            """)
            
            # Get index definitions
            index_defs = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes 
                WHERE schemaname = 'mem0_vectors'
            """)
            
            # Combine data
            size_map = {(row['schemaname'], row['indexname']): row['size_bytes'] 
                       for row in index_sizes}
            def_map = {(row['schemaname'], row['indexname']): row['indexdef'] 
                      for row in index_defs}
            
            analyses = []
            for row in index_stats:
                schema = row['schemaname']
                index_name = row['indexname']
                table_name = row['tablename']
                
                size_bytes = size_map.get((schema, index_name), 0)
                size_mb = size_bytes / (1024 * 1024)
                index_def = def_map.get((schema, index_name), '')
                
                # Determine index type
                index_type = self._determine_index_type(index_def)
                
                # Calculate efficiency score
                efficiency = self._calculate_index_efficiency(row, size_bytes)
                
                # Generate recommendations
                recommendations = self._generate_index_recommendations(
                    row, size_bytes, index_type
                )
                
                analyses.append(IndexAnalysis(
                    index_name=index_name,
                    table_name=table_name,
                    index_type=index_type,
                    size_mb=size_mb,
                    usage_count=row['usage_count'] or 0,
                    last_used=None,  # Would need additional tracking
                    efficiency_score=efficiency,
                    recommendations=recommendations
                ))
            
            return analyses
    
    def _determine_index_type(self, index_def: str) -> str:
        """Determine index type from definition"""
        index_def_lower = index_def.lower()
        
        if 'using hnsw' in index_def_lower:
            return 'HNSW'
        elif 'using ivfflat' in index_def_lower:
            return 'IVFFlat'
        elif 'using gin' in index_def_lower:
            return 'GIN'
        elif 'using btree' in index_def_lower:
            return 'B-Tree'
        elif 'using hash' in index_def_lower:
            return 'Hash'
        else:
            return 'Unknown'
    
    def _calculate_index_efficiency(self, stats: Dict, size_bytes: int) -> float:
        """Calculate index efficiency score (0-1)"""
        usage_count = stats['usage_count'] or 0
        tuples_read = stats['idx_tup_read'] or 0
        tuples_fetched = stats['idx_tup_fetch'] or 0
        
        # Factors for efficiency
        usage_factor = min(usage_count / 1000, 1.0)  # Normalize to 1000 uses
        fetch_ratio = tuples_fetched / max(tuples_read, 1)  # How many tuples actually used
        size_factor = max(0, 1 - (size_bytes / (100 * 1024 * 1024)))  # Penalty for large size
        
        efficiency = (usage_factor * 0.4 + fetch_ratio * 0.4 + size_factor * 0.2)
        return min(max(efficiency, 0), 1)
    
    def _generate_index_recommendations(self, stats: Dict, size_bytes: int, 
                                      index_type: str) -> List[str]:
        """Generate recommendations for index optimization"""
        recommendations = []
        usage_count = stats['usage_count'] or 0
        
        if usage_count == 0:
            recommendations.append("Consider dropping this unused index")
        elif usage_count < 10:
            recommendations.append("Low usage - monitor if index is necessary")
        
        if size_bytes > 100 * 1024 * 1024:  # 100MB
            recommendations.append("Large index - consider partial indexing or compression")
        
        if index_type == 'HNSW':
            recommendations.append("Consider tuning m and ef_construction parameters")
        elif index_type == 'IVFFlat':
            recommendations.append("Consider adjusting lists parameter based on data size")
        
        return recommendations
    
    async def optimize_database_settings(self) -> Dict[str, Any]:
        """Analyze and optimize database settings"""
        async with self.pool.acquire() as conn:
            # Get current settings
            current_settings = await conn.fetch("""
                SELECT name, setting, unit, context, short_desc
                FROM pg_settings 
                WHERE name IN (
                    'shared_buffers', 'work_mem', 'maintenance_work_mem',
                    'effective_cache_size', 'random_page_cost', 'seq_page_cost',
                    'max_parallel_workers', 'max_parallel_workers_per_gather'
                )
                ORDER BY name
            """)
            
            # Get database statistics
            db_stats = await conn.fetchrow("""
                SELECT 
                    pg_size_pretty(pg_database_size(current_database())) as db_size,
                    (SELECT count(*) FROM mem0_vectors.memories) as memory_count,
                    (SELECT avg(array_length(embedding, 1)) FROM mem0_vectors.memories 
                     WHERE embedding IS NOT NULL) as avg_vector_dim
            """)
            
            # Memory information
            memory_info = psutil.virtual_memory()
            
            recommendations = []
            
            # Analyze each setting
            for setting in current_settings:
                name = setting['name']
                current_value = setting['setting']
                unit = setting['unit']
                
                if name == 'shared_buffers':
                    # Should be 25% of RAM
                    recommended_mb = int(memory_info.total / (1024 * 1024) * 0.25)
                    current_mb = self._parse_memory_setting(current_value, unit)
                    
                    if current_mb < recommended_mb * 0.8:
                        recommendations.append(
                            f"Increase shared_buffers to ~{recommended_mb}MB (currently {current_mb}MB)"
                        )
                
                elif name == 'work_mem':
                    # Should be reasonable for sorts and joins
                    recommended_mb = max(64, int(memory_info.total / (1024 * 1024) * 0.05))
                    current_mb = self._parse_memory_setting(current_value, unit)
                    
                    if current_mb < recommended_mb * 0.5:
                        recommendations.append(
                            f"Consider increasing work_mem to {recommended_mb}MB for better sort performance"
                        )
                
                elif name == 'effective_cache_size':
                    # Should be 75% of total RAM
                    recommended_mb = int(memory_info.total / (1024 * 1024) * 0.75)
                    current_mb = self._parse_memory_setting(current_value, unit)
                    
                    if current_mb < recommended_mb * 0.8:
                        recommendations.append(
                            f"Increase effective_cache_size to {recommended_mb}MB for better query planning"
                        )
            
            # Vector-specific recommendations
            memory_count = db_stats['memory_count'] or 0
            if memory_count > 10000:
                recommendations.append("Consider using HNSW indexes for large vector datasets")
            
            if memory_count > 100000:
                recommendations.append("Consider partitioning memories table by user_id or date")
            
            return {
                'current_settings': [dict(row) for row in current_settings],
                'database_stats': dict(db_stats),
                'system_memory_gb': memory_info.total / (1024**3),
                'recommendations': recommendations
            }
    
    def _parse_memory_setting(self, value: str, unit: str) -> int:
        """Parse PostgreSQL memory setting to MB"""
        try:
            value = int(value)
            if unit == 'kB':
                return value // 1024
            elif unit == 'MB':
                return value
            elif unit == 'GB':
                return value * 1024
            elif unit == '8kB':
                return (value * 8) // 1024
            else:
                return value // 1024  # Assume kB
        except:
            return 0
    
    async def generate_performance_report(self, user_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'database_settings': await self.optimize_database_settings(),
            'index_analysis': [asdict(idx) for idx in await self.analyze_index_performance()],
            'recommendations': []
        }
        
        # Add similarity search analysis if user_id provided
        if user_id:
            # Generate a test embedding for analysis
            test_embedding = np.random.rand(1536).tolist()
            similarity_analysis = await self.optimize_similarity_search(user_id, test_embedding)
            report['similarity_search_analysis'] = similarity_analysis
        
        # Generate overall recommendations
        overall_recommendations = []
        
        # Index recommendations
        unused_indexes = [idx for idx in report['index_analysis'] if idx['usage_count'] == 0]
        if unused_indexes:
            overall_recommendations.append(
                f"Found {len(unused_indexes)} unused indexes that could be dropped"
            )
        
        low_efficiency_indexes = [idx for idx in report['index_analysis'] 
                                 if idx['efficiency_score'] < 0.3]
        if low_efficiency_indexes:
            overall_recommendations.append(
                f"Found {len(low_efficiency_indexes)} low-efficiency indexes that need optimization"
            )
        
        # Database setting recommendations
        overall_recommendations.extend(report['database_settings']['recommendations'])
        
        report['overall_recommendations'] = overall_recommendations
        
        return report

# Example usage and CLI
async def main():
    """Test the query performance tuner"""
    DB_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/mem0ai')
    
    config = QueryPerformanceConfig(
        work_mem_mb=256,
        maintenance_work_mem_mb=512,
        enable_query_logging=True
    )
    
    optimizer = PostgreSQLOptimizer(DB_URL, config)
    
    try:
        await optimizer.initialize()
        
        print("Generating performance report...")
        report = await optimizer.generate_performance_report("test_user")
        
        print("\n=== DATABASE PERFORMANCE REPORT ===")
        print(f"Generated: {report['timestamp']}")
        
        print(f"\nDatabase Settings Analysis:")
        settings = report['database_settings']
        print(f"  Database size: {settings.get('database_stats', {}).get('db_size', 'Unknown')}")
        print(f"  Memory count: {settings.get('database_stats', {}).get('memory_count', 0)}")
        print(f"  System RAM: {settings.get('system_memory_gb', 0):.1f} GB")
        
        print(f"\nIndex Analysis ({len(report['index_analysis'])} indexes):")
        for idx in report['index_analysis'][:5]:  # Show top 5
            print(f"  {idx['index_name']} ({idx['index_type']}): "
                  f"{idx['usage_count']} uses, {idx['size_mb']:.1f}MB, "
                  f"efficiency: {idx['efficiency_score']:.2f}")
        
        print(f"\nOverall Recommendations ({len(report['overall_recommendations'])}):")
        for i, rec in enumerate(report['overall_recommendations'][:5], 1):
            print(f"  {i}. {rec}")
        
        if 'similarity_search_analysis' in report:
            print(f"\nSimilarity Search Analysis:")
            search_analysis = report['similarity_search_analysis']
            for query_name, metrics in search_analysis.items():
                if 'execution_time_ms' in metrics:
                    print(f"  {query_name}: {metrics['execution_time_ms']:.2f}ms, "
                          f"index: {metrics.get('index_used', 'none')}")
        
    finally:
        await optimizer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())