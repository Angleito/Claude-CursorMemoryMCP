"""Advanced rate limiting and DDoS protection."""

import hashlib
import json
import time
from collections import defaultdict

import redis.asyncio as redis
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from slowapi import Limiter
from slowapi.util import get_remote_address

from auth.models import User
from config.settings import ROLE_RATE_LIMITS
from config.settings import get_settings
from monitoring.audit_logger import audit_logger

settings = get_settings()

# Rate limiting constants
DDOS_THRESHOLD = 1000  # requests per minute
DDOS_WINDOW_SECONDS = 60
ANON_REQUESTS_LIMIT = 20
ANON_WINDOW_SECONDS = 60
MIN_TOKENS_REQUIRED = 1
DEFAULT_MAX_CONCURRENT = 10
CONCURRENT_TIMEOUT_SECONDS = 30
DDOS_BLOCK_DURATION_HOURS = 1
DDOS_BLOCK_SECONDS = 3600  # 1 hour
SUSPICIOUS_USER_AGENT_SCORE = 30
INVALID_USER_AGENT_SCORE = 20
SUSPICIOUS_HEADER_SCORE = 40
SUSPICION_THRESHOLD = 50
SUSPICIOUS_BLOCK_SECONDS = 300  # 5 minutes
MIN_USER_AGENT_LENGTH = 10
RETRY_AFTER_SECONDS = 60
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_MINUTES = 30
LOGIN_WINDOW_MINUTES = 15
LOGIN_LOCKOUT_SECONDS = 1800  # 30 minutes
LOGIN_WINDOW_SECONDS = 900  # 15 minutes


class AdvancedRateLimiter:
    """Advanced rate limiting with multiple strategies."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.settings = settings

        # Suspicious activity tracking
        self.suspicious_ips = set()
        self.failed_attempts = defaultdict(int)

        # DDoS detection thresholds
        self.ddos_threshold = DDOS_THRESHOLD
        self.ddos_window = DDOS_WINDOW_SECONDS

    async def check_rate_limit(
        self,
        request: Request,
        user: User | None = None,
        custom_limit: dict | None = None,
    ) -> tuple[bool, dict[str, int]]:
        """Check rate limit for request.

        Returns (is_allowed, rate_limit_info).
        """
        # Get identifier and limits
        identifier = self._get_identifier(request, user)
        limits = custom_limit or self._get_rate_limits(user)

        # Check multiple rate limiting strategies
        checks = [
            await self._check_token_bucket(identifier, limits),
            await self._check_sliding_window(identifier, limits),
            await self._check_concurrent_requests(identifier),
            await self._check_ddos_protection(request.client.host),
            await self._check_suspicious_activity(request),
        ]

        # All checks must pass
        is_allowed = all(check[0] for check in checks)

        # Combine rate limit info
        rate_limit_info = {}
        for _allowed, info in checks:
            rate_limit_info.update(info)

        if not is_allowed:
            await self._log_rate_limit_violation(request, user, rate_limit_info)

        return is_allowed, rate_limit_info

    def _get_identifier(self, request: Request, user: User | None) -> str:
        """Get unique identifier for rate limiting."""
        if user:
            return f"user:{user.id}"

        # Use IP address for anonymous users
        ip = request.client.host

        # Consider X-Forwarded-For for proxy setups
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            ip = forwarded_for.split(",")[0].strip()

        return f"ip:{ip}"

    def _get_rate_limits(self, user: User | None) -> dict[str, int]:
        """Get rate limits based on user role."""
        if user:
            return ROLE_RATE_LIMITS.get(user.role.value, ROLE_RATE_LIMITS["user"])

        # Stricter limits for anonymous users
        return {"requests": ANON_REQUESTS_LIMIT, "window": ANON_WINDOW_SECONDS}

    async def _check_token_bucket(
        self, identifier: str, limits: dict[str, int]
    ) -> tuple[bool, dict[str, int]]:
        """Token bucket algorithm for burst protection."""
        bucket_key = f"bucket:{identifier}"
        capacity = limits["requests"]
        refill_rate = capacity / limits["window"]  # tokens per second

        # Get current bucket state
        bucket_data = await self.redis.get(bucket_key)

        now = time.time()

        if bucket_data:
            data = json.loads(bucket_data)
            tokens = data["tokens"]
            last_refill = data["last_refill"]
        else:
            tokens = capacity
            last_refill = now

        # Refill tokens
        time_passed = now - last_refill
        tokens = min(capacity, tokens + (time_passed * refill_rate))

        # Check if request can be processed
        if tokens >= MIN_TOKENS_REQUIRED:
            tokens -= MIN_TOKENS_REQUIRED
            allowed = True
        else:
            allowed = False

        # Update bucket
        bucket_data = {"tokens": tokens, "last_refill": now}
        await self.redis.setex(bucket_key, limits["window"], json.dumps(bucket_data))

        return allowed, {"bucket_tokens": int(tokens), "bucket_capacity": capacity}

    async def _check_sliding_window(
        self, identifier: str, limits: dict[str, int]
    ) -> tuple[bool, dict[str, int]]:
        """Sliding window rate limiting."""
        window_key = f"window:{identifier}"
        window_size = limits["window"]
        max_requests = limits["requests"]

        now = time.time()
        window_start = now - window_size

        # Clean old entries and count current requests
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(window_key, 0, window_start)
        pipe.zcard(window_key)
        pipe.zadd(window_key, {str(now): now})
        pipe.expire(window_key, window_size)

        results = await pipe.execute()
        current_requests = results[1]

        allowed = current_requests < max_requests

        return allowed, {
            "window_requests": current_requests,
            "window_limit": max_requests,
            "window_reset": int(now + window_size),
        }

    async def _check_concurrent_requests(
        self, identifier: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT
    ) -> tuple[bool, dict[str, int]]:
        """Check concurrent request limits."""
        concurrent_key = f"concurrent:{identifier}"

        # Get current concurrent requests
        current = await self.redis.get(concurrent_key)
        current_count = int(current) if current else 0

        allowed = current_count < max_concurrent

        if allowed:
            # Increment counter with short TTL
            await self.redis.incr(concurrent_key)
            await self.redis.expire(concurrent_key, CONCURRENT_TIMEOUT_SECONDS)

        return allowed, {
            "concurrent_requests": current_count,
            "concurrent_limit": max_concurrent,
        }

    async def _check_ddos_protection(self, ip: str) -> tuple[bool, dict[str, int]]:
        """DDoS protection based on request patterns."""
        ddos_key = f"ddos:{ip}"

        # Count requests in the last minute
        pipe = self.redis.pipeline()
        pipe.incr(ddos_key)
        pipe.expire(ddos_key, self.ddos_window)

        results = await pipe.execute()
        request_count = results[0]

        # Check if threshold exceeded
        if request_count > self.ddos_threshold:
            # Add to suspicious IPs
            self.suspicious_ips.add(ip)

            # Block for longer period
            block_key = f"blocked:{ip}"
            await self.redis.setex(block_key, DDOS_BLOCK_SECONDS, "ddos_protection")

            return False, {
                "ddos_blocked": True,
                "ddos_requests": request_count,
                "ddos_threshold": self.ddos_threshold,
            }

        return True, {
            "ddos_requests": request_count,
            "ddos_threshold": self.ddos_threshold,
        }

    async def _check_suspicious_activity(
        self, request: Request
    ) -> tuple[bool, dict[str, int]]:
        """Check for suspicious request patterns."""
        ip = request.client.host
        user_agent = request.headers.get("User-Agent", "")

        suspicion_score = 0
        reasons = []

        # Check for blocked IP
        block_key = f"blocked:{ip}"
        if await self.redis.exists(block_key):
            return False, {"suspicious_blocked": True}

        # Suspicious user agents
        suspicious_agents = [
            "bot",
            "crawler",
            "spider",
            "scraper",
            "scanner",
            "curl",
            "wget",
            "python-requests",
        ]

        if any(agent in user_agent.lower() for agent in suspicious_agents):
            suspicion_score += SUSPICIOUS_USER_AGENT_SCORE
            reasons.append("suspicious_user_agent")

        # Missing or invalid user agent
        if not user_agent or len(user_agent) < MIN_USER_AGENT_LENGTH:
            suspicion_score += INVALID_USER_AGENT_SCORE
            reasons.append("invalid_user_agent")

        # Check for common attack patterns in headers
        for header_name, header_value in request.headers.items():
            if self._is_suspicious_header(header_name, header_value):
                suspicion_score += SUSPICIOUS_HEADER_SCORE
                reasons.append(f"suspicious_header_{header_name}")

        # High suspicion score blocks request
        if suspicion_score >= SUSPICION_THRESHOLD:
            # Temporary block
            block_key = f"suspicious:{ip}"
            await self.redis.setex(
                block_key, SUSPICIOUS_BLOCK_SECONDS, "suspicious_activity"
            )

            return False, {
                "suspicious_blocked": True,
                "suspicion_score": suspicion_score,
                "reasons": reasons,
            }

        return True, {"suspicion_score": suspicion_score, "reasons": reasons}

    def _is_suspicious_header(self, name: str, value: str) -> bool:
        """Check if header contains suspicious content."""
        suspicious_patterns = [
            "script",
            "javascript",
            "vbscript",
            "onload",
            "onerror",
            "<",
            ">",
            "eval(",
            "alert(",
            "document.cookie",
            "union",
            "select",
            "insert",
            "delete",
            "drop",
            "exec",
        ]

        combined = f"{name}:{value}".lower()
        return any(pattern in combined for pattern in suspicious_patterns)

    async def _log_rate_limit_violation(
        self, request: Request, user: User | None, rate_limit_info: dict
    ):
        """Log rate limit violations."""
        await audit_logger.log_security_event(
            event_type="rate_limit_exceeded",
            user_id=user.id if user else None,
            ip_address=request.client.host,
            details={
                "path": str(request.url),
                "method": request.method,
                "user_agent": request.headers.get("User-Agent", ""),
                "rate_limit_info": rate_limit_info,
            },
            severity="medium",
        )

    async def decrement_concurrent_requests(self, identifier: str):
        """Decrement concurrent request counter (call this when request completes)."""
        concurrent_key = f"concurrent:{identifier}"
        await self.redis.decr(concurrent_key)

    async def get_rate_limit_status(
        self, identifier: str, limits: dict[str, int]
    ) -> dict[str, int]:
        """Get current rate limit status."""
        # Get sliding window count
        window_key = f"window:{identifier}"
        window_size = limits["window"]
        now = time.time()
        window_start = now - window_size

        await self.redis.zremrangebyscore(window_key, 0, window_start)
        current_requests = await self.redis.zcard(window_key)

        return {
            "limit": limits["requests"],
            "remaining": max(0, limits["requests"] - current_requests),
            "reset": int(now + window_size),
            "window": window_size,
        }


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, redis_client: redis.Redis):
        self.app = app
        self.limiter = AdvancedRateLimiter(redis_client)

    async def __call__(self, request: Request, call_next):

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Get current user if available
        user = getattr(request.state, "current_user", None)

        # Check rate limits
        is_allowed, rate_limit_info = await self.limiter.check_rate_limit(request, user)

        if not is_allowed:
            # Determine appropriate error message
            if rate_limit_info.get("ddos_blocked"):
                detail = "Request blocked due to DDoS protection"
                status_code = status.HTTP_429_TOO_MANY_REQUESTS
            elif rate_limit_info.get("suspicious_blocked"):
                detail = "Request blocked due to suspicious activity"
                status_code = status.HTTP_403_FORBIDDEN
            else:
                detail = "Rate limit exceeded"
                status_code = status.HTTP_429_TOO_MANY_REQUESTS

            raise HTTPException(
                status_code=status_code,
                detail=detail,
                headers={
                    "X-RateLimit-Limit": str(rate_limit_info.get("window_limit", 0)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_limit_info.get("window_reset", 0)),
                    "Retry-After": str(RETRY_AFTER_SECONDS),
                },
            )

        # Add rate limit headers
        response = await call_next(request)

        # Get current status for headers
        identifier = self.limiter._get_identifier(request, user)
        limits = self.limiter._get_rate_limits(user)
        status_info = await self.limiter.get_rate_limit_status(identifier, limits)

        response.headers["X-RateLimit-Limit"] = str(status_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(status_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(status_info["reset"])

        # Decrement concurrent request counter
        await self.limiter.decrement_concurrent_requests(identifier)

        return response


# Simple rate limiter for specific endpoints
limiter = Limiter(key_func=get_remote_address)


# Rate limiting decorators
def rate_limit(requests: str):
    """Rate limiting decorator.

    Usage: @rate_limit("10/minute").
    """
    return limiter.limit(requests)


# IP-based rate limiting for login attempts
class LoginRateLimiter:
    """Specialized rate limiter for login attempts."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.max_attempts = MAX_LOGIN_ATTEMPTS
        self.lockout_duration = LOGIN_LOCKOUT_SECONDS
        self.window = LOGIN_WINDOW_SECONDS

    async def check_login_attempts(self, ip: str, email: str) -> bool:
        """Check if login attempts exceed limit."""
        # Check both IP and email-based limits
        ip_key = f"login_attempts:ip:{ip}"
        email_key = f"login_attempts:email:{hashlib.sha256(email.encode()).hexdigest()}"

        ip_attempts = await self.redis.get(ip_key)
        email_attempts = await self.redis.get(email_key)

        ip_count = int(ip_attempts) if ip_attempts else 0
        email_count = int(email_attempts) if email_attempts else 0

        return ip_count < self.max_attempts and email_count < self.max_attempts

    async def record_failed_attempt(self, ip: str, email: str):
        """Record a failed login attempt."""
        ip_key = f"login_attempts:ip:{ip}"
        email_key = f"login_attempts:email:{hashlib.sha256(email.encode()).hexdigest()}"

        # Increment counters
        pipe = self.redis.pipeline()
        pipe.incr(ip_key)
        pipe.expire(ip_key, self.window)
        pipe.incr(email_key)
        pipe.expire(email_key, self.window)

        results = await pipe.execute()

        # Check if lockout needed
        if results[0] >= self.max_attempts or results[2] >= self.max_attempts:
            # Add to blocked list
            block_key = f"blocked_login:{ip}"
            await self.redis.setex(block_key, self.lockout_duration, "login_attempts")

    async def clear_attempts(self, ip: str, email: str):
        """Clear login attempts after successful login."""
        ip_key = f"login_attempts:ip:{ip}"
        email_key = f"login_attempts:email:{hashlib.sha256(email.encode()).hexdigest()}"

        await self.redis.delete(ip_key, email_key)


# Global rate limiter instances
def get_rate_limiter(redis_client: redis.Redis) -> AdvancedRateLimiter:
    return AdvancedRateLimiter(redis_client)


def get_login_rate_limiter(redis_client: redis.Redis) -> LoginRateLimiter:
    return LoginRateLimiter(redis_client)
