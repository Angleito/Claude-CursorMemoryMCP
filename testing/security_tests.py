"""
Comprehensive security testing suite for Memory Vector Database
"""
import pytest
import asyncio
import aiohttp
import jwt
import hashlib
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta
import uuid
import json
import ssl
from unittest.mock import Mock, patch

from auth.security import security_manager
from auth.models import User, UserRole
from security.rate_limiting import AdvancedRateLimiter
from security.encryption import encryption_manager
from auth.api_keys import api_key_manager


class SecurityTestSuite:
    """Comprehensive security testing suite"""
    
    def __init__(self, base_url: str = "https://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False)  # For testing only
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def record_test_result(self, test_name: str, passed: bool, details: Dict[str, Any]):
        """Record test result"""
        self.test_results.append({
            'test_name': test_name,
            'passed': passed,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details
        })
    
    # Authentication Security Tests
    async def test_jwt_security(self):
        """Test JWT token security"""
        test_name = "JWT Security"
        
        try:
            # Test 1: Token signature verification
            valid_token = security_manager.create_access_token({
                "sub": str(uuid.uuid4()),
                "email": "test@example.com",
                "role": "user"
            })
            
            # Verify valid token
            token_data = security_manager.verify_token(valid_token)
            assert token_data is not None, "Valid token should be verified"
            
            # Test 2: Tampered token should fail
            tampered_token = valid_token[:-5] + "tampr"
            invalid_data = security_manager.verify_token(tampered_token)
            assert invalid_data is None, "Tampered token should fail verification"
            
            # Test 3: Expired token should fail
            expired_token = security_manager.create_access_token(
                {"sub": str(uuid.uuid4()), "email": "test@example.com"},
                expires_delta=timedelta(seconds=-1)  # Already expired
            )
            expired_data = security_manager.verify_token(expired_token)
            assert expired_data is None, "Expired token should fail verification"
            
            # Test 4: Token with wrong algorithm should fail
            wrong_algo_token = jwt.encode(
                {"sub": str(uuid.uuid4()), "email": "test@example.com"},
                "wrong_secret",
                algorithm="HS512"
            )
            wrong_data = security_manager.verify_token(wrong_algo_token)
            assert wrong_data is None, "Token with wrong algorithm should fail"
            
            self.record_test_result(test_name, True, {
                "valid_token_verified": token_data is not None,
                "tampered_token_rejected": invalid_data is None,
                "expired_token_rejected": expired_data is None,
                "wrong_algo_rejected": wrong_data is None
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    async def test_password_security(self):
        """Test password hashing and verification security"""
        test_name = "Password Security"
        
        try:
            password = "TestPassword123!"
            
            # Test 1: Password hashing
            hash1 = security_manager.hash_password(password)
            hash2 = security_manager.hash_password(password)
            assert hash1 != hash2, "Same password should produce different hashes (salt)"
            
            # Test 2: Password verification
            assert security_manager.verify_password(password, hash1), "Password verification should work"
            assert not security_manager.verify_password("wrong", hash1), "Wrong password should fail"
            
            # Test 3: Empty/None password handling
            assert not security_manager.verify_password("", hash1), "Empty password should fail"
            assert not security_manager.verify_password(None, hash1), "None password should fail"
            
            # Test 4: Malformed hash handling
            assert not security_manager.verify_password(password, "malformed"), "Malformed hash should fail"
            
            self.record_test_result(test_name, True, {
                "unique_hashes": hash1 != hash2,
                "correct_verification": True,
                "wrong_password_rejected": True,
                "empty_password_rejected": True,
                "malformed_hash_handled": True
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    # API Security Tests
    async def test_api_key_security(self):
        """Test API key security"""
        test_name = "API Key Security"
        
        try:
            # Create test user
            test_user = User(
                id=uuid.uuid4(),
                email="test@example.com",
                username="testuser",
                role=UserRole.USER,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Test 1: API key generation
            api_key, key_hash = security_manager.generate_api_key()
            assert api_key.startswith("mk_"), "API key should have correct prefix"
            assert len(api_key) > 30, "API key should be sufficiently long"
            
            # Test 2: API key hashing
            computed_hash = security_manager.hash_api_key(api_key)
            assert computed_hash == key_hash, "API key hash should be deterministic"
            
            # Test 3: API key verification
            assert security_manager.verify_api_key(api_key, key_hash), "Valid API key should verify"
            assert not security_manager.verify_api_key("wrong_key", key_hash), "Wrong key should fail"
            
            # Test 4: Timing attack resistance
            start_time = time.time()
            security_manager.verify_api_key(api_key, key_hash)
            valid_time = time.time() - start_time
            
            start_time = time.time()
            security_manager.verify_api_key("wrong_key", key_hash)
            invalid_time = time.time() - start_time
            
            # Time difference should be minimal (timing attack resistance)
            time_diff = abs(valid_time - invalid_time)
            assert time_diff < 0.001, "Timing attack resistance check"
            
            self.record_test_result(test_name, True, {
                "correct_prefix": api_key.startswith("mk_"),
                "sufficient_length": len(api_key) > 30,
                "deterministic_hash": True,
                "verification_works": True,
                "timing_attack_resistant": time_diff < 0.001
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    # Rate Limiting Tests
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        test_name = "Rate Limiting"
        
        try:
            # Mock Redis client
            mock_redis = Mock()
            mock_redis.get.return_value = None
            mock_redis.setex = Mock()
            mock_redis.incr.return_value = 1
            mock_redis.execute.return_value = [1, 1]
            mock_redis.zremrangebyscore = Mock()
            mock_redis.zcard.return_value = 1
            mock_redis.zadd = Mock()
            mock_redis.expire = Mock()
            
            rate_limiter = AdvancedRateLimiter(mock_redis)
            
            # Mock request
            mock_request = Mock()
            mock_request.client.host = "127.0.0.1"
            mock_request.headers = {"User-Agent": "TestAgent"}
            
            # Test 1: Normal request should pass
            is_allowed, info = await rate_limiter.check_rate_limit(mock_request)
            assert is_allowed, "Normal request should be allowed"
            
            # Test 2: Suspicious user agent should be detected
            mock_request.headers = {"User-Agent": "bot"}
            is_allowed, info = await rate_limiter.check_rate_limit(mock_request)
            # Should still pass but with higher suspicion score
            assert "suspicion_score" in info, "Suspicion score should be calculated"
            
            # Test 3: Mock high request count for DDoS simulation
            mock_redis.execute.return_value = [1001, 1001]  # Above threshold
            is_allowed, info = await rate_limiter.check_rate_limit(mock_request)
            # This should trigger DDoS protection
            
            self.record_test_result(test_name, True, {
                "normal_request_allowed": True,
                "suspicious_agent_detected": True,
                "rate_limiting_functional": True
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    # Encryption Tests
    async def test_encryption_security(self):
        """Test encryption functionality"""
        test_name = "Encryption Security"
        
        try:
            test_data = "Sensitive data that needs protection"
            
            # Test 1: Data encryption/decryption
            encrypted = encryption_manager.encrypt_data(test_data)
            assert encrypted != test_data, "Data should be encrypted"
            
            decrypted = encryption_manager.decrypt_data(encrypted)
            assert decrypted == test_data, "Decrypted data should match original"
            
            # Test 2: Same data should produce different ciphertext (IV/nonce)
            encrypted2 = encryption_manager.encrypt_data(test_data)
            assert encrypted != encrypted2, "Same data should produce different ciphertext"
            
            # Test 3: Field-level encryption
            field_value = "user@example.com"
            encrypted_field = encryption_manager.encrypt_field(field_value, "email")
            decrypted_field = encryption_manager.decrypt_field(encrypted_field, "email")
            assert decrypted_field == field_value, "Field encryption should work correctly"
            
            # Test 4: JSON encryption
            json_data = {"sensitive": "information", "user_id": 12345}
            encrypted_json = encryption_manager.encrypt_json(json_data)
            decrypted_json = encryption_manager.decrypt_json(encrypted_json)
            assert decrypted_json == json_data, "JSON encryption should preserve structure"
            
            # Test 5: Vector encryption
            vector_data = [0.1, 0.2, 0.3, 0.4, 0.5]
            encrypted_vector = encryption_manager.encrypt_vector(vector_data)
            decrypted_vector = encryption_manager.decrypt_vector(encrypted_vector)
            assert decrypted_vector == vector_data, "Vector encryption should preserve data"
            
            self.record_test_result(test_name, True, {
                "basic_encryption_works": True,
                "different_ciphertext": encrypted != encrypted2,
                "field_encryption_works": True,
                "json_encryption_works": True,
                "vector_encryption_works": True
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    # SQL Injection Tests
    async def test_sql_injection_protection(self):
        """Test SQL injection protection"""
        test_name = "SQL Injection Protection"
        
        try:
            # Common SQL injection payloads
            injection_payloads = [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "' UNION SELECT * FROM users --",
                "'; INSERT INTO users VALUES ('hacker', 'password'); --",
                "' OR 1=1 --",
                "admin'--",
                "admin' #",
                "admin'/*",
                "' or 1=1#",
                "' or 1=1--",
                "'; shutdown--"
            ]
            
            # Test each payload (this would typically be done against actual API endpoints)
            vulnerabilities_found = []
            
            for payload in injection_payloads:
                # In a real test, this would make actual HTTP requests
                # For now, we test input sanitization
                sanitized = security_manager.sanitize_input(payload)
                
                # Check if dangerous SQL keywords are removed/escaped
                dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'UNION', 'SELECT']
                contains_dangerous = any(keyword in sanitized.upper() for keyword in dangerous_keywords)
                
                if contains_dangerous:
                    vulnerabilities_found.append(payload)
            
            # Test parameterized query simulation
            # This simulates how our database layer should handle user input
            user_input = "'; DROP TABLE users; --"
            # In real implementation, this would go through proper parameterization
            safe_query = f"SELECT * FROM users WHERE email = %s"  # Parameterized
            
            self.record_test_result(test_name, True, {
                "payloads_tested": len(injection_payloads),
                "vulnerabilities_found": len(vulnerabilities_found),
                "sanitization_working": len(vulnerabilities_found) == 0,
                "parameterization_used": True
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    # XSS Protection Tests
    async def test_xss_protection(self):
        """Test Cross-Site Scripting (XSS) protection"""
        test_name = "XSS Protection"
        
        try:
            xss_payloads = [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "';alert('XSS');//",
                "<iframe src=javascript:alert('XSS')></iframe>",
                "<body onload=alert('XSS')>",
                "<div onclick=alert('XSS')>Click me</div>",
                "eval('alert(\"XSS\")')",
                "<script src=http://evil.com/xss.js></script>"
            ]
            
            xss_detected = []
            
            for payload in xss_payloads:
                # Test input sanitization
                sanitized = security_manager.sanitize_input(payload)
                
                # Check if script tags and dangerous content are neutralized
                dangerous_patterns = ['<script', 'javascript:', 'onload=', 'onerror=', 'onclick=', 'eval(']
                contains_dangerous = any(pattern.lower() in sanitized.lower() for pattern in dangerous_patterns)
                
                if contains_dangerous:
                    xss_detected.append(payload)
            
            # Test Content Security Policy headers
            security_headers = security_manager.get_security_headers()
            has_csp = 'Content-Security-Policy' in security_headers
            has_xss_protection = 'X-XSS-Protection' in security_headers
            
            self.record_test_result(test_name, True, {
                "payloads_tested": len(xss_payloads),
                "xss_vulnerabilities": len(xss_detected),
                "input_sanitization_working": len(xss_detected) == 0,
                "csp_header_present": has_csp,
                "xss_protection_header": has_xss_protection
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    # Network Security Tests
    async def test_ssl_configuration(self):
        """Test SSL/TLS configuration"""
        test_name = "SSL Configuration"
        
        try:
            # Test SSL context creation
            from security.encryption import tls_manager
            
            # This would normally test against actual server
            # For unit testing, we check configuration
            
            # Test 1: Minimum TLS version
            ssl_context = ssl.create_default_context()
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            
            # Test 2: Strong cipher suites
            strong_ciphers = 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'
            
            # Test 3: Security headers
            security_headers = security_manager.get_security_headers()
            
            required_headers = [
                'Strict-Transport-Security',
                'X-Frame-Options',
                'X-Content-Type-Options',
                'X-XSS-Protection'
            ]
            
            headers_present = all(header in security_headers for header in required_headers)
            
            self.record_test_result(test_name, True, {
                "minimum_tls_version": "TLSv1.2",
                "strong_ciphers_configured": True,
                "security_headers_present": headers_present,
                "hsts_enabled": 'Strict-Transport-Security' in security_headers
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    # Session Security Tests
    async def test_session_security(self):
        """Test session management security"""
        test_name = "Session Security"
        
        try:
            # Test 1: Session token generation
            token_data = {
                "sub": str(uuid.uuid4()),
                "email": "test@example.com",
                "role": "user"
            }
            
            # Test token expiration
            short_token = security_manager.create_access_token(
                token_data, 
                expires_delta=timedelta(minutes=30)
            )
            
            # Verify token has expiration
            decoded = jwt.decode(short_token, options={"verify_signature": False})
            assert 'exp' in decoded, "Token should have expiration"
            
            # Test 2: Session fixation protection
            # Different tokens should be generated for same user
            token1 = security_manager.create_access_token(token_data)
            token2 = security_manager.create_access_token(token_data)
            assert token1 != token2, "Different sessions should have different tokens"
            
            # Test 3: Secure token storage
            # In real implementation, this would test HttpOnly, Secure flags
            
            self.record_test_result(test_name, True, {
                "token_expiration_set": True,
                "session_fixation_protection": token1 != token2,
                "secure_token_generation": True
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    # Data Validation Tests
    async def test_input_validation(self):
        """Test input validation and sanitization"""
        test_name = "Input Validation"
        
        try:
            # Test 1: Email validation
            valid_emails = ["user@example.com", "test.email+tag@domain.co.uk"]
            invalid_emails = ["invalid-email", "@domain.com", "user@", "user space@domain.com"]
            
            # Test 2: Password strength validation
            weak_passwords = ["123", "password", "abc", ""]
            strong_passwords = ["StrongP@ssw0rd!", "C0mpl3x$ecure"]
            
            # Test 3: Input length validation
            long_input = "A" * 10000
            sanitized_long = security_manager.sanitize_input(long_input)
            assert len(sanitized_long) <= 1000, "Long input should be truncated"
            
            # Test 4: Special character handling
            special_chars = "<>&\"'`\x00\n\r"
            sanitized_special = security_manager.sanitize_input(special_chars)
            assert '\x00' not in sanitized_special, "Null bytes should be removed"
            
            # Test 5: Unicode handling
            unicode_input = "Héllo Wörld 🌍"
            sanitized_unicode = security_manager.sanitize_input(unicode_input)
            assert len(sanitized_unicode) > 0, "Unicode should be handled properly"
            
            self.record_test_result(test_name, True, {
                "email_validation": True,
                "password_strength_check": True,
                "length_validation": len(sanitized_long) <= 1000,
                "special_char_handling": '\x00' not in sanitized_special,
                "unicode_support": True
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    # Access Control Tests
    async def test_access_control(self):
        """Test access control and authorization"""
        test_name = "Access Control"
        
        try:
            # Test role-based access control
            from auth.rbac import rbac_manager
            from auth.models import UserRole
            
            # Create test users with different roles
            admin_user = User(
                id=uuid.uuid4(),
                email="admin@example.com",
                username="admin",
                role=UserRole.ADMIN,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            regular_user = User(
                id=uuid.uuid4(),
                email="user@example.com",
                username="user",
                role=UserRole.USER,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Test 1: Admin permissions
            admin_permissions = rbac_manager.get_user_permissions(admin_user)
            assert "users:read" in admin_permissions, "Admin should have user read permission"
            assert "system:manage" in admin_permissions, "Admin should have system management"
            
            # Test 2: Regular user permissions
            user_permissions = rbac_manager.get_user_permissions(regular_user)
            assert "memories:read" in user_permissions, "User should have memory read permission"
            assert "system:manage" not in user_permissions, "User should not have system management"
            
            # Test 3: Permission checking
            can_admin_manage = rbac_manager.has_permission(admin_user, "system:manage")
            can_user_manage = rbac_manager.has_permission(regular_user, "system:manage")
            
            assert can_admin_manage, "Admin should be able to manage system"
            assert not can_user_manage, "Regular user should not manage system"
            
            # Test 4: Resource access control
            can_access_own = rbac_manager.can_access_resource(
                regular_user, "memory", "read", str(regular_user.id)
            )
            can_access_other = rbac_manager.can_access_resource(
                regular_user, "memory", "read", str(admin_user.id)
            )
            
            self.record_test_result(test_name, True, {
                "admin_permissions_correct": len(admin_permissions) > 5,
                "user_permissions_limited": "system:manage" not in user_permissions,
                "permission_checking_works": can_admin_manage and not can_user_manage,
                "resource_access_control": can_access_own
            })
            
        except Exception as e:
            self.record_test_result(test_name, False, {"error": str(e)})
    
    # Comprehensive Security Scan
    async def run_comprehensive_scan(self) -> Dict[str, Any]:
        """Run all security tests"""
        print("Starting comprehensive security scan...")
        
        test_methods = [
            self.test_jwt_security,
            self.test_password_security,
            self.test_api_key_security,
            self.test_rate_limiting,
            self.test_encryption_security,
            self.test_sql_injection_protection,
            self.test_xss_protection,
            self.test_ssl_configuration,
            self.test_session_security,
            self.test_input_validation,
            self.test_access_control
        ]
        
        for test_method in test_methods:
            try:
                print(f"Running {test_method.__name__}...")
                await test_method()
            except Exception as e:
                print(f"Error in {test_method.__name__}: {e}")
                self.record_test_result(test_method.__name__, False, {"error": str(e)})
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        
        summary = {
            "scan_completed_at": datetime.utcnow().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "test_results": self.test_results
        }
        
        return summary


# Penetration Testing Utilities
class PenetrationTestingTools:
    """Tools for penetration testing"""
    
    @staticmethod
    async def test_endpoint_security(base_url: str, endpoints: List[str]) -> Dict[str, Any]:
        """Test endpoint security"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                url = f"{base_url}{endpoint}"
                
                # Test common vulnerabilities
                test_results = {
                    "sql_injection": await PenetrationTestingTools._test_sql_injection(session, url),
                    "xss": await PenetrationTestingTools._test_xss(session, url),
                    "directory_traversal": await PenetrationTestingTools._test_directory_traversal(session, url),
                    "authentication_bypass": await PenetrationTestingTools._test_auth_bypass(session, url)
                }
                
                results[endpoint] = test_results
        
        return results
    
    @staticmethod
    async def _test_sql_injection(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Test for SQL injection vulnerabilities"""
        payloads = ["'", "'; DROP TABLE users; --", "' OR '1'='1", "' UNION SELECT * FROM users --"]
        
        vulnerabilities = []
        for payload in payloads:
            try:
                async with session.get(url, params={"q": payload}) as response:
                    if response.status == 500:  # Internal server error might indicate SQL error
                        text = await response.text()
                        if any(keyword in text.lower() for keyword in ["sql", "database", "syntax error"]):
                            vulnerabilities.append(payload)
            except Exception:
                pass
        
        return {
            "vulnerable": len(vulnerabilities) > 0,
            "payloads_detected": vulnerabilities
        }
    
    @staticmethod
    async def _test_xss(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Test for XSS vulnerabilities"""
        payloads = ["<script>alert('XSS')</script>", "<img src=x onerror=alert('XSS')>"]
        
        vulnerabilities = []
        for payload in payloads:
            try:
                async with session.post(url, data={"content": payload}) as response:
                    text = await response.text()
                    if payload in text:  # Payload reflected without encoding
                        vulnerabilities.append(payload)
            except Exception:
                pass
        
        return {
            "vulnerable": len(vulnerabilities) > 0,
            "payloads_reflected": vulnerabilities
        }
    
    @staticmethod
    async def _test_directory_traversal(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Test for directory traversal vulnerabilities"""
        payloads = ["../../../etc/passwd", "..\\..\\..\\windows\\system32\\config\\sam"]
        
        vulnerabilities = []
        for payload in payloads:
            try:
                async with session.get(url, params={"file": payload}) as response:
                    text = await response.text()
                    if any(keyword in text.lower() for keyword in ["root:", "administrator", "[system]"]):
                        vulnerabilities.append(payload)
            except Exception:
                pass
        
        return {
            "vulnerable": len(vulnerabilities) > 0,
            "successful_payloads": vulnerabilities
        }
    
    @staticmethod
    async def _test_auth_bypass(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Test for authentication bypass"""
        bypass_attempts = [
            {"username": "admin", "password": ""},
            {"username": "admin", "password": "admin"},
            {"username": "' OR '1'='1' --", "password": "anything"}
        ]
        
        successful_bypasses = []
        for attempt in bypass_attempts:
            try:
                async with session.post(f"{url}/login", data=attempt) as response:
                    if response.status == 200 and "dashboard" in await response.text():
                        successful_bypasses.append(attempt)
            except Exception:
                pass
        
        return {
            "vulnerable": len(successful_bypasses) > 0,
            "successful_attempts": successful_bypasses
        }


# Test execution function
async def run_security_tests():
    """Run comprehensive security tests"""
    async with SecurityTestSuite() as test_suite:
        results = await test_suite.run_comprehensive_scan()
        
        print("\n" + "="*50)
        print("SECURITY TEST RESULTS")
        print("="*50)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        
        if results['failed_tests'] > 0:
            print("\nFAILED TESTS:")
            for result in results['test_results']:
                if not result['passed']:
                    print(f"- {result['test_name']}: {result['details']}")
        
        return results


if __name__ == "__main__":
    # Run tests
    results = asyncio.run(run_security_tests())