#!/usr/bin/env python3
"""
Production-grade Embedding Generation Pipeline for mem0ai
Supports multiple embedding models with optimized batch processing and caching
"""

import asyncio
import aiohttp
import asyncpg
import numpy as np
import hashlib
import pickle
import logging
import time
import json
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import tiktoken
import redis.asyncio as redis
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import openai
from openai import AsyncOpenAI
import cohere
import anthropic
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    COHERE = "cohere"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"
    ANTHROPIC = "anthropic"

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    provider: EmbeddingProvider
    model_name: str
    dimensions: int
    max_tokens: int
    batch_size: int = 100
    rate_limit_rpm: int = 1000
    cache_ttl_hours: int = 24
    retry_attempts: int = 3
    retry_delay: float = 1.0
    api_key: Optional[str] = None
    
    @classmethod
    def get_default_configs(cls) -> Dict[str, 'EmbeddingConfig']:
        """Get default configurations for different providers"""
        return {
            'openai_ada002': cls(
                provider=EmbeddingProvider.OPENAI,
                model_name='text-embedding-ada-002',
                dimensions=1536,
                max_tokens=8191,
                batch_size=100,
                rate_limit_rpm=3000
            ),
            'openai_3_small': cls(
                provider=EmbeddingProvider.OPENAI,
                model_name='text-embedding-3-small',
                dimensions=1536,
                max_tokens=8191,
                batch_size=100,
                rate_limit_rpm=3000
            ),
            'openai_3_large': cls(
                provider=EmbeddingProvider.OPENAI,
                model_name='text-embedding-3-large',
                dimensions=3072,
                max_tokens=8191,
                batch_size=100,
                rate_limit_rpm=3000
            ),
            'cohere_v3': cls(
                provider=EmbeddingProvider.COHERE,
                model_name='embed-english-v3.0',
                dimensions=1024,
                max_tokens=512,
                batch_size=96,
                rate_limit_rpm=1000
            ),
            'sentence_bert': cls(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
                model_name='all-MiniLM-L6-v2',
                dimensions=384,
                max_tokens=512,
                batch_size=32,
                rate_limit_rpm=10000  # Local model, no API limits
            ),
            'sentence_mpnet': cls(
                provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
                model_name='all-mpnet-base-v2',
                dimensions=768,
                max_tokens=512,
                batch_size=32,
                rate_limit_rpm=10000
            ),
            'huggingface_gte': cls(
                provider=EmbeddingProvider.HUGGINGFACE,
                model_name='thenlper/gte-large',
                dimensions=1024,
                max_tokens=512,
                batch_size=16,
                rate_limit_rpm=10000
            )
        }

@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    text: str
    user_id: str
    memory_id: Optional[str] = None
    metadata: Optional[Dict] = None
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    text: str
    embedding: List[float]
    model_name: str
    dimensions: int
    processing_time_ms: float
    token_count: int
    cache_hit: bool
    timestamp: datetime
    user_id: str
    memory_id: Optional[str] = None
    metadata: Optional[Dict] = None

class RateLimiter:
    """Async rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: float = 60.0):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit slot"""
        async with self.lock:
            now = time.time()
            # Remove old calls outside time window
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                # Calculate wait time
                oldest_call = min(self.calls)
                wait_time = self.time_window - (now - oldest_call) + 0.1
                await asyncio.sleep(wait_time)
                return await self.acquire()
            
            self.calls.append(now)

class EmbeddingCache:
    """Redis-based caching for embeddings"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.redis_url)
        
    async def get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model"""
        combined = f"{model_name}:{text}"
        return f"embedding:{hashlib.sha256(combined.encode()).hexdigest()}"
        
    async def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding"""
        if not self.redis_client:
            return None
            
        try:
            key = await self.get_cache_key(text, model_name)
            cached_data = await self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
        
    async def set(self, text: str, model_name: str, embedding: List[float], 
                  ttl_hours: int = 24):
        """Cache embedding"""
        if not self.redis_client:
            return
            
        try:
            key = await self.get_cache_key(text, model_name)
            data = pickle.dumps(embedding)
            await self.redis_client.setex(key, timedelta(hours=ttl_hours), data)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            
    async def clear_expired(self):
        """Clear expired cache entries (handled by Redis TTL)"""
        pass
        
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

class BaseEmbeddingProvider:
    """Base class for embedding providers"""
    
    def __init__(self, config: EmbeddingConfig, cache: EmbeddingCache):
        self.config = config
        self.cache = cache
        self.rate_limiter = RateLimiter(config.rate_limit_rpm)
        
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts"""
        raise NotImplementedError
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            encoding = tiktoken.encoding_for_model(self.config.model_name)
            return len(encoding.encode(text))
        except:
            # Fallback to approximate token count
            return len(text.split()) * 1.3
            
    def truncate_text(self, text: str) -> str:
        """Truncate text to model's max tokens"""
        token_count = self.count_tokens(text)
        if token_count <= self.config.max_tokens:
            return text
            
        # Approximate truncation
        ratio = self.config.max_tokens / token_count
        words = text.split()
        truncated_words = words[:int(len(words) * ratio * 0.9)]  # Safety margin
        return ' '.join(truncated_words)

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, config: EmbeddingConfig, cache: EmbeddingCache):
        super().__init__(config, cache)
        self.client = AsyncOpenAI(api_key=config.api_key or os.getenv('OPENAI_API_KEY'))
        
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        await self.rate_limiter.acquire()
        
        # Truncate texts
        truncated_texts = [self.truncate_text(text) for text in texts]
        
        try:
            response = await self.client.embeddings.create(
                model=self.config.model_name,
                input=truncated_texts
            )
            
            embeddings = []
            for data in response.data:
                embeddings.append(data.embedding)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embedding provider"""
    
    def __init__(self, config: EmbeddingConfig, cache: EmbeddingCache):
        super().__init__(config, cache)
        self.client = cohere.AsyncClient(config.api_key or os.getenv('COHERE_API_KEY'))
        
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere API"""
        await self.rate_limiter.acquire()
        
        # Truncate texts
        truncated_texts = [self.truncate_text(text) for text in texts]
        
        try:
            response = await self.client.embed(
                texts=truncated_texts,
                model=self.config.model_name,
                input_type="search_document"
            )
            
            return response.embeddings
            
        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            raise

class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Sentence Transformers local embedding provider"""
    
    def __init__(self, config: EmbeddingConfig, cache: EmbeddingCache):
        super().__init__(config, cache)
        self.model = None
        self.lock = threading.Lock()
        
    def _load_model(self):
        """Load model (thread-safe)"""
        if self.model is None:
            with self.lock:
                if self.model is None:
                    self.model = SentenceTransformer(self.config.model_name)
                    if torch.cuda.is_available():
                        self.model = self.model.cuda()
                        
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformers"""
        self._load_model()
        
        # Truncate texts
        truncated_texts = [self.truncate_text(text) for text in texts]
        
        try:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                embeddings = await loop.run_in_executor(
                    executor, 
                    self.model.encode, 
                    truncated_texts
                )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"SentenceTransformer embedding error: {e}")
            raise

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace transformers embedding provider"""
    
    def __init__(self, config: EmbeddingConfig, cache: EmbeddingCache):
        super().__init__(config, cache)
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()
        
    def _load_model(self):
        """Load model and tokenizer (thread-safe)"""
        if self.model is None:
            with self.lock:
                if self.model is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                    self.model = AutoModel.from_pretrained(self.config.model_name)
                    if torch.cuda.is_available():
                        self.model = self.model.cuda()
                        
    async def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode batch of texts"""
        self._load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=self.config.max_tokens,
            return_tensors='pt'
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token or mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy().tolist()
        
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using HuggingFace transformers"""
        try:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                embeddings = await loop.run_in_executor(
                    executor, 
                    lambda: asyncio.run(self._encode_batch(texts))
                )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise

class EmbeddingPipeline:
    """Main embedding generation pipeline"""
    
    def __init__(self, db_url: str, redis_url: str = "redis://localhost:6379"):
        self.db_url = db_url
        self.cache = EmbeddingCache(redis_url)
        self.db_pool = None
        self.providers = {}
        self.default_config = None
        
    async def initialize(self):
        """Initialize pipeline"""
        # Initialize cache
        await self.cache.initialize()
        
        # Initialize database pool
        self.db_pool = await asyncpg.create_pool(
            self.db_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # Load default configurations
        configs = EmbeddingConfig.get_default_configs()
        
        # Initialize providers
        for name, config in configs.items():
            try:
                if config.provider == EmbeddingProvider.OPENAI:
                    self.providers[name] = OpenAIEmbeddingProvider(config, self.cache)
                elif config.provider == EmbeddingProvider.COHERE:
                    self.providers[name] = CohereEmbeddingProvider(config, self.cache)
                elif config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                    self.providers[name] = SentenceTransformerProvider(config, self.cache)
                elif config.provider == EmbeddingProvider.HUGGINGFACE:
                    self.providers[name] = HuggingFaceEmbeddingProvider(config, self.cache)
                    
                logger.info(f"Initialized provider: {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize provider {name}: {e}")
        
        # Set default provider
        if 'openai_ada002' in self.providers:
            self.default_config = configs['openai_ada002']
        elif 'sentence_bert' in self.providers:
            self.default_config = configs['sentence_bert']
        else:
            raise RuntimeError("No embedding providers available")
            
        logger.info("Embedding pipeline initialized")
        
    async def cleanup(self):
        """Cleanup resources"""
        await self.cache.close()
        if self.db_pool:
            await self.db_pool.close()
            
    async def generate_embedding(self, request: EmbeddingRequest, 
                               provider_name: Optional[str] = None) -> EmbeddingResult:
        """Generate single embedding"""
        results = await self.generate_embeddings([request], provider_name)
        return results[0]
        
    async def generate_embeddings(self, requests: List[EmbeddingRequest],
                                provider_name: Optional[str] = None) -> List[EmbeddingResult]:
        """Generate embeddings for multiple requests"""
        if not requests:
            return []
            
        # Select provider
        if provider_name and provider_name in self.providers:
            provider = self.providers[provider_name]
            config = provider.config
        else:
            provider_name = list(self.providers.keys())[0]
            provider = self.providers[provider_name]
            config = self.default_config
            
        # Group requests by cache status
        cached_results = {}
        uncached_requests = []
        
        for i, request in enumerate(requests):
            # Check cache
            cached_embedding = await self.cache.get(request.text, config.model_name)
            if cached_embedding:
                cached_results[i] = EmbeddingResult(
                    text=request.text,
                    embedding=cached_embedding,
                    model_name=config.model_name,
                    dimensions=config.dimensions,
                    processing_time_ms=0,
                    token_count=provider.count_tokens(request.text),
                    cache_hit=True,
                    timestamp=datetime.now(),
                    user_id=request.user_id,
                    memory_id=request.memory_id,
                    metadata=request.metadata
                )
            else:
                uncached_requests.append((i, request))
        
        # Process uncached requests in batches
        results = [None] * len(requests)
        
        # Fill cached results
        for i, result in cached_results.items():
            results[i] = result
            
        # Process uncached in batches
        if uncached_requests:
            batch_size = config.batch_size
            for i in range(0, len(uncached_requests), batch_size):
                batch = uncached_requests[i:i + batch_size]
                batch_texts = [req.text for _, req in batch]
                
                start_time = time.time()
                
                try:
                    embeddings = await provider.generate_embeddings(batch_texts)
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Create results and cache embeddings
                    for j, ((orig_idx, request), embedding) in enumerate(zip(batch, embeddings)):
                        result = EmbeddingResult(
                            text=request.text,
                            embedding=embedding,
                            model_name=config.model_name,
                            dimensions=config.dimensions,
                            processing_time_ms=processing_time / len(batch),
                            token_count=provider.count_tokens(request.text),
                            cache_hit=False,
                            timestamp=datetime.now(),
                            user_id=request.user_id,
                            memory_id=request.memory_id,
                            metadata=request.metadata
                        )
                        
                        results[orig_idx] = result
                        
                        # Cache the embedding
                        await self.cache.set(
                            request.text, 
                            config.model_name, 
                            embedding,
                            config.cache_ttl_hours
                        )
                        
                except Exception as e:
                    logger.error(f"Batch embedding generation failed: {e}")
                    # Create error results
                    for orig_idx, request in batch:
                        results[orig_idx] = EmbeddingResult(
                            text=request.text,
                            embedding=[0.0] * config.dimensions,
                            model_name=config.model_name,
                            dimensions=config.dimensions,
                            processing_time_ms=0,
                            token_count=0,
                            cache_hit=False,
                            timestamp=datetime.now(),
                            user_id=request.user_id,
                            memory_id=request.memory_id,
                            metadata={'error': str(e)}
                        )
        
        return results
    
    async def store_embeddings(self, results: List[EmbeddingResult]) -> List[str]:
        """Store embeddings in database"""
        stored_ids = []
        
        async with self.db_pool.acquire() as conn:
            for result in results:
                if result.memory_id:
                    # Update existing memory
                    await conn.execute("""
                        UPDATE mem0_vectors.memories 
                        SET embedding = $1, updated_at = NOW()
                        WHERE id = $2
                    """, result.embedding, result.memory_id)
                    stored_ids.append(result.memory_id)
                else:
                    # Insert new memory
                    memory_id = await conn.fetchval("""
                        INSERT INTO mem0_vectors.memories 
                        (user_id, memory_text, embedding, metadata)
                        VALUES ($1, $2, $3, $4)
                        RETURNING id
                    """, result.user_id, result.text, result.embedding, result.metadata or {})
                    stored_ids.append(memory_id)
                    
        return stored_ids
    
    async def get_available_providers(self) -> Dict[str, Dict]:
        """Get information about available providers"""
        provider_info = {}
        for name, provider in self.providers.items():
            config = provider.config
            provider_info[name] = {
                'provider': config.provider.value,
                'model_name': config.model_name,
                'dimensions': config.dimensions,
                'max_tokens': config.max_tokens,
                'batch_size': config.batch_size
            }
        return provider_info
    
    async def benchmark_providers(self, test_texts: List[str]) -> Dict[str, Dict]:
        """Benchmark all available providers"""
        benchmark_results = {}
        
        for provider_name in self.providers.keys():
            start_time = time.time()
            
            try:
                requests = [
                    EmbeddingRequest(text=text, user_id="benchmark") 
                    for text in test_texts
                ]
                
                results = await self.generate_embeddings(requests, provider_name)
                
                total_time = (time.time() - start_time) * 1000
                cache_hits = sum(1 for r in results if r.cache_hit)
                
                benchmark_results[provider_name] = {
                    'total_time_ms': total_time,
                    'avg_time_per_text_ms': total_time / len(test_texts),
                    'cache_hits': cache_hits,
                    'cache_miss': len(test_texts) - cache_hits,
                    'success_rate': len([r for r in results if 'error' not in (r.metadata or {})]) / len(results),
                    'dimensions': results[0].dimensions if results else 0
                }
                
            except Exception as e:
                benchmark_results[provider_name] = {
                    'error': str(e),
                    'total_time_ms': float('inf'),
                    'success_rate': 0.0
                }
                
        return benchmark_results

# Example usage and CLI interface
async def main():
    """Main function for testing the pipeline"""
    import sys
    
    # Database configuration
    DB_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/mem0ai')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline(DB_URL, REDIS_URL)
    
    try:
        await pipeline.initialize()
        
        # Test texts
        test_texts = [
            "I love programming in Python",
            "Machine learning is fascinating",
            "Vector databases are powerful",
            "Embeddings capture semantic meaning",
            "Natural language processing enables AI"
        ]
        
        # Create requests
        requests = [
            EmbeddingRequest(text=text, user_id="test_user")
            for text in test_texts
        ]
        
        # Test different providers
        providers = await pipeline.get_available_providers()
        print(f"Available providers: {list(providers.keys())}")
        
        # Benchmark providers
        print("\nBenchmarking providers...")
        benchmark_results = await pipeline.benchmark_providers(test_texts)
        
        for provider, metrics in benchmark_results.items():
            print(f"\n{provider}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
                
        # Generate embeddings
        print(f"\nGenerating embeddings for {len(test_texts)} texts...")
        results = await pipeline.generate_embeddings(requests)
        
        for i, result in enumerate(results):
            print(f"Text {i+1}: {result.text[:50]}...")
            print(f"  Dimensions: {result.dimensions}")
            print(f"  Processing time: {result.processing_time_ms:.2f}ms")
            print(f"  Cache hit: {result.cache_hit}")
            print(f"  Tokens: {result.token_count}")
            
        # Store embeddings
        stored_ids = await pipeline.store_embeddings(results)
        print(f"\nStored {len(stored_ids)} embeddings in database")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    finally:
        await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())