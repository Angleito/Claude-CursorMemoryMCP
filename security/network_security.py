"""Network security implementations including firewall, intrusion detection, and traffic analysis."""

import asyncio
import ipaddress
import json
import logging
import re
import socket
import ssl
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import psutil
from fastapi import Request
from pydantic import BaseModel, Field

from config.settings import get_settings
from monitoring.audit_logger import audit_logger

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Represents a network security event."""
    
    event_type: str
    source_ip: str
    destination_ip: str
    port: int
    timestamp: datetime
    severity: str
    details: Dict[str, Any]
    blocked: bool = False


@dataclass
class FirewallRule:
    """Represents a firewall rule."""
    
    rule_id: str
    action: str  # allow, deny, drop
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    port: Optional[int] = None
    protocol: str = "tcp"
    description: str = ""
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ThreatIntelligence:
    """Threat intelligence and IP reputation management."""
    
    def __init__(self):
        self.malicious_ips: Set[str] = set()
        self.suspicious_ips: Set[str] = set()
        self.tor_exit_nodes: Set[str] = set()
        self.vpn_ranges: Set[str] = set()
        self.last_update: Optional[datetime] = None
        
    async def update_threat_feeds(self) -> None:
        """Update threat intelligence feeds from external sources."""
        try:
            # Update malicious IP feeds
            await self._update_malicious_ips()
            
            # Update Tor exit nodes
            await self._update_tor_nodes()
            
            # Update VPN ranges
            await self._update_vpn_ranges()
            
            self.last_update = datetime.utcnow()
            logger.info("Threat intelligence feeds updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update threat feeds: {e}")
            await audit_logger.log_security_event(
                event_type="threat_feed_update_failed",
                details={"error": str(e)},
                severity="medium"
            )
    
    async def _update_malicious_ips(self) -> None:
        """Update malicious IP database from threat feeds."""
        try:
            # Example threat feed URLs (replace with actual feeds)
            feeds = [
                "https://feeds.firehol.org/ipsets/firehol_level1.netset",
                "https://reputation.alienvault.com/reputation.data"
            ]
            
            new_ips = set()
            
            async with aiohttp.ClientSession() as session:
                for feed_url in feeds:
                    try:
                        async with session.get(feed_url, timeout=30) as response:
                            if response.status == 200:
                                content = await response.text()
                                # Parse IP addresses from feed
                                ips = self._parse_ip_feed(content)
                                new_ips.update(ips)
                    except Exception as e:
                        logger.warning(f"Failed to fetch threat feed {feed_url}: {e}")
            
            self.malicious_ips = new_ips
            logger.info(f"Updated {len(new_ips)} malicious IPs")
            
        except Exception as e:
            logger.error(f"Failed to update malicious IPs: {e}")
    
    async def _update_tor_nodes(self) -> None:
        """Update Tor exit node list."""
        try:
            url = "https://check.torproject.org/torbulkexitlist"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        tor_ips = set(line.strip() for line in content.split('\n') 
                                    if line.strip() and not line.startswith('#'))
                        self.tor_exit_nodes = tor_ips
                        logger.info(f"Updated {len(tor_ips)} Tor exit nodes")
                        
        except Exception as e:
            logger.error(f"Failed to update Tor nodes: {e}")
    
    async def _update_vpn_ranges(self) -> None:
        """Update known VPN IP ranges."""
        try:
            # This would typically come from commercial VPN detection services
            # For now, we'll use some common VPN provider ranges
            known_vpn_ranges = [
                "185.159.156.0/22",  # NordVPN
                "103.231.88.0/23",   # ExpressVPN
                "141.98.80.0/20",    # Surfshark
            ]
            
            self.vpn_ranges = set(known_vpn_ranges)
            logger.info(f"Updated {len(known_vpn_ranges)} VPN ranges")
            
        except Exception as e:
            logger.error(f"Failed to update VPN ranges: {e}")
    
    def _parse_ip_feed(self, content: str) -> Set[str]:
        """Parse IP addresses from threat feed content."""
        ips = set()
        
        # Regular expression to match IP addresses
        ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            # Extract IP addresses
            matches = ip_pattern.findall(line)
            for ip in matches:
                try:
                    # Validate IP address
                    ipaddress.ip_address(ip)
                    ips.add(ip)
                except ValueError:
                    continue
        
        return ips
    
    def is_malicious_ip(self, ip: str) -> bool:
        """Check if IP is known to be malicious."""
        return ip in self.malicious_ips
    
    def is_tor_exit_node(self, ip: str) -> bool:
        """Check if IP is a Tor exit node."""
        return ip in self.tor_exit_nodes
    
    def is_vpn_ip(self, ip: str) -> bool:
        """Check if IP belongs to a VPN provider."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            for range_str in self.vpn_ranges:
                network = ipaddress.ip_network(range_str)
                if ip_obj in network:
                    return True
        except ValueError:
            pass
        return False
    
    def get_ip_reputation(self, ip: str) -> Dict[str, Any]:
        """Get comprehensive IP reputation information."""
        return {
            "ip": ip,
            "is_malicious": self.is_malicious_ip(ip),
            "is_tor": self.is_tor_exit_node(ip),
            "is_vpn": self.is_vpn_ip(ip),
            "risk_score": self._calculate_risk_score(ip),
            "last_updated": self.last_update.isoformat() if self.last_update else None
        }
    
    def _calculate_risk_score(self, ip: str) -> int:
        """Calculate risk score for IP address (0-100)."""
        score = 0
        
        if self.is_malicious_ip(ip):
            score += 80
        if self.is_tor_exit_node(ip):
            score += 30
        if self.is_vpn_ip(ip):
            score += 20
            
        return min(score, 100)


class IntrusionDetectionSystem:
    """Network intrusion detection and prevention system."""
    
    def __init__(self):
        self.threat_intel = ThreatIntelligence()
        self.connection_tracking: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: List[Dict[str, Any]] = []
        self.load_attack_patterns()
        
    def load_attack_patterns(self) -> None:
        """Load known attack patterns for detection."""
        self.suspicious_patterns = [
            {
                "name": "SQL Injection",
                "pattern": r"(?i)(union.*select|insert.*into|delete.*from|drop.*table|exec.*sp_|xp_cmdshell)",
                "severity": "high"
            },
            {
                "name": "XSS Attempt",
                "pattern": r"(?i)(<script|javascript:|vbscript:|onload=|onerror=|eval\()",
                "severity": "medium"
            },
            {
                "name": "Directory Traversal",
                "pattern": r"(\.\./|\.\.\|%2e%2e%2f|%252e%252e%252f)",
                "severity": "medium"
            },
            {
                "name": "Command Injection",
                "pattern": r"(?i)(;|\||&|`|\$\(|%0a|%0d|nc\s+-|wget\s+|curl\s+)",
                "severity": "high"
            },
            {
                "name": "File Inclusion",
                "pattern": r"(?i)(php://|file://|ftp://|data:|expect://|zip://)",
                "severity": "high"
            }
        ]
    
    async def analyze_request(self, request: Request) -> SecurityEvent:
        """Analyze incoming request for security threats."""
        client_ip = self._get_client_ip(request)
        
        # Basic request analysis
        event = SecurityEvent(
            event_type="request_analysis",
            source_ip=client_ip,
            destination_ip=self._get_server_ip(),
            port=request.url.port or 443,
            timestamp=datetime.utcnow(),
            severity="info",
            details={
                "method": request.method,
                "path": str(request.url.path),
                "user_agent": request.headers.get("User-Agent", ""),
                "referer": request.headers.get("Referer", "")
            }
        )
        
        # Check IP reputation
        reputation = self.threat_intel.get_ip_reputation(client_ip)
        event.details["ip_reputation"] = reputation
        
        if reputation["is_malicious"]:
            event.event_type = "malicious_ip_detected"
            event.severity = "critical"
            event.blocked = True
            
        # Analyze request for attack patterns
        await self._analyze_attack_patterns(request, event)
        
        # Analyze request frequency (potential DoS)
        await self._analyze_request_frequency(client_ip, event)
        
        # Track connection
        self._track_connection(client_ip, event)
        
        return event
    
    async def _analyze_attack_patterns(self, request: Request, event: SecurityEvent) -> None:
        """Analyze request for known attack patterns."""
        request_data = f"{request.url.path}?{request.url.query}"
        
        # Check headers for suspicious content
        for header_name, header_value in request.headers.items():
            request_data += f"{header_name}:{header_value} "
        
        # Check for attack patterns
        detected_attacks = []
        
        for pattern_info in self.suspicious_patterns:
            pattern = pattern_info["pattern"]
            if re.search(pattern, request_data):
                detected_attacks.append({
                    "attack_type": pattern_info["name"],
                    "severity": pattern_info["severity"]
                })
                
                # Update event severity
                if pattern_info["severity"] == "critical":
                    event.severity = "critical"
                    event.blocked = True
                elif pattern_info["severity"] == "high" and event.severity not in ["critical"]:
                    event.severity = "high"
                elif pattern_info["severity"] == "medium" and event.severity in ["info", "low"]:
                    event.severity = "medium"
        
        if detected_attacks:
            event.event_type = "attack_pattern_detected"
            event.details["detected_attacks"] = detected_attacks
    
    async def _analyze_request_frequency(self, client_ip: str, event: SecurityEvent) -> None:
        """Analyze request frequency for DoS detection."""
        now = time.time()
        window = 60  # 1 minute window
        
        # Get recent requests from this IP
        recent_requests = self.connection_tracking[client_ip]
        
        # Count requests in the last minute
        recent_count = sum(1 for req_time in recent_requests if now - req_time < window)
        
        # Thresholds for different severity levels
        if recent_count > 1000:  # 1000 requests per minute
            event.event_type = "dos_attack_detected"
            event.severity = "critical"
            event.blocked = True
        elif recent_count > 500:  # 500 requests per minute
            event.event_type = "high_frequency_requests"
            event.severity = "high"
        elif recent_count > 100:  # 100 requests per minute
            event.event_type = "elevated_request_frequency"
            event.severity = "medium"
        
        event.details["request_frequency"] = recent_count
    
    def _track_connection(self, client_ip: str, event: SecurityEvent) -> None:
        """Track connection for pattern analysis."""
        self.connection_tracking[client_ip].append(time.time())
        
        # Block IP if critical event detected
        if event.severity == "critical" and event.blocked:
            self.blocked_ips.add(client_ip)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host
    
    def _get_server_ip(self) -> str:
        """Get server IP address."""
        try:
            # Get the server's IP address
            hostname = socket.gethostname()
            return socket.gethostbyname(hostname)
        except Exception:
            return "127.0.0.1"
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked."""
        return ip in self.blocked_ips
    
    def block_ip(self, ip: str, reason: str) -> None:
        """Block an IP address."""
        self.blocked_ips.add(ip)
        logger.warning(f"Blocked IP {ip}: {reason}")
    
    def unblock_ip(self, ip: str) -> None:
        """Unblock an IP address."""
        self.blocked_ips.discard(ip)
        logger.info(f"Unblocked IP {ip}")


class NetworkFirewall:
    """Software firewall implementation."""
    
    def __init__(self):
        self.rules: List[FirewallRule] = []
        self.default_action = "deny"
        self.load_default_rules()
    
    def load_default_rules(self) -> None:
        """Load default firewall rules."""
        default_rules = [
            FirewallRule(
                rule_id="allow_http",
                action="allow",
                port=80,
                protocol="tcp",
                description="Allow HTTP traffic"
            ),
            FirewallRule(
                rule_id="allow_https",
                action="allow",
                port=443,
                protocol="tcp",
                description="Allow HTTPS traffic"
            ),
            FirewallRule(
                rule_id="allow_ssh",
                action="allow",
                port=22,
                protocol="tcp",
                description="Allow SSH (admin access only)"
            ),
            FirewallRule(
                rule_id="deny_all_others",
                action="deny",
                description="Deny all other traffic by default"
            )
        ]
        
        self.rules.extend(default_rules)
    
    def add_rule(self, rule: FirewallRule) -> None:
        """Add a new firewall rule."""
        self.rules.append(rule)
        logger.info(f"Added firewall rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a firewall rule by ID."""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                del self.rules[i]
                logger.info(f"Removed firewall rule: {rule_id}")
                return True
        return False
    
    def check_connection(self, source_ip: str, dest_ip: str, port: int, protocol: str = "tcp") -> str:
        """Check if connection is allowed by firewall rules."""
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            # Check if rule matches
            if self._rule_matches(rule, source_ip, dest_ip, port, protocol):
                return rule.action
        
        return self.default_action
    
    def _rule_matches(self, rule: FirewallRule, source_ip: str, dest_ip: str, port: int, protocol: str) -> bool:
        """Check if a rule matches the connection parameters."""
        # Check protocol
        if rule.protocol != "any" and rule.protocol != protocol:
            return False
        
        # Check source IP
        if rule.source_ip and not self._ip_matches(source_ip, rule.source_ip):
            return False
        
        # Check destination IP
        if rule.destination_ip and not self._ip_matches(dest_ip, rule.destination_ip):
            return False
        
        # Check port
        if rule.port and rule.port != port:
            return False
        
        return True
    
    def _ip_matches(self, ip: str, rule_ip: str) -> bool:
        """Check if IP matches rule (supports CIDR notation)."""
        try:
            if "/" in rule_ip:
                # CIDR notation
                network = ipaddress.ip_network(rule_ip, strict=False)
                return ipaddress.ip_address(ip) in network
            else:
                # Exact match
                return ip == rule_ip
        except ValueError:
            return False
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all firewall rules."""
        return [
            {
                "rule_id": rule.rule_id,
                "action": rule.action,
                "source_ip": rule.source_ip,
                "destination_ip": rule.destination_ip,
                "port": rule.port,
                "protocol": rule.protocol,
                "description": rule.description,
                "enabled": rule.enabled,
                "created_at": rule.created_at.isoformat()
            }
            for rule in self.rules
        ]


class SSLSecurityChecker:
    """SSL/TLS security configuration checker."""
    
    def __init__(self):
        self.weak_ciphers = [
            "RC4",
            "DES",
            "3DES",
            "MD5",
            "SHA1"
        ]
        
        self.weak_protocols = [
            "SSLv2",
            "SSLv3",
            "TLSv1.0",
            "TLSv1.1"
        ]
    
    async def check_ssl_configuration(self, hostname: str, port: int = 443) -> Dict[str, Any]:
        """Check SSL/TLS configuration of a host."""
        result = {
            "hostname": hostname,
            "port": port,
            "timestamp": datetime.utcnow().isoformat(),
            "certificate_info": {},
            "protocol_support": {},
            "cipher_suites": [],
            "vulnerabilities": [],
            "security_score": 0
        }
        
        try:
            # Check certificate
            cert_info = await self._check_certificate(hostname, port)
            result["certificate_info"] = cert_info
            
            # Check protocol support
            protocol_support = await self._check_protocol_support(hostname, port)
            result["protocol_support"] = protocol_support
            
            # Check cipher suites
            cipher_suites = await self._check_cipher_suites(hostname, port)
            result["cipher_suites"] = cipher_suites
            
            # Identify vulnerabilities
            vulnerabilities = self._identify_vulnerabilities(cert_info, protocol_support, cipher_suites)
            result["vulnerabilities"] = vulnerabilities
            
            # Calculate security score
            security_score = self._calculate_security_score(cert_info, protocol_support, cipher_suites, vulnerabilities)
            result["security_score"] = security_score
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to check SSL configuration for {hostname}:{port}: {e}")
        
        return result
    
    async def _check_certificate(self, hostname: str, port: int) -> Dict[str, Any]:
        """Check SSL certificate details."""
        cert_info = {}
        
        try:
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    cert_info = {
                        "subject": dict(x[0] for x in cert.get("subject", [])),
                        "issuer": dict(x[0] for x in cert.get("issuer", [])),
                        "version": cert.get("version"),
                        "serial_number": cert.get("serialNumber"),
                        "not_before": cert.get("notBefore"),
                        "not_after": cert.get("notAfter"),
                        "signature_algorithm": cert.get("signatureAlgorithm"),
                        "san": cert.get("subjectAltName", [])
                    }
                    
                    # Check if certificate is expired or expiring soon
                    not_after = datetime.strptime(cert.get("notAfter"), "%b %d %H:%M:%S %Y %Z")
                    days_until_expiry = (not_after - datetime.utcnow()).days
                    cert_info["days_until_expiry"] = days_until_expiry
                    cert_info["is_expired"] = days_until_expiry < 0
                    cert_info["expires_soon"] = days_until_expiry < 30
                    
        except Exception as e:
            cert_info["error"] = str(e)
        
        return cert_info
    
    async def _check_protocol_support(self, hostname: str, port: int) -> Dict[str, bool]:
        """Check which SSL/TLS protocols are supported."""
        protocols = {
            "SSLv2": ssl.PROTOCOL_SSLv23,  # This will be rejected by modern OpenSSL
            "SSLv3": ssl.PROTOCOL_SSLv23,
            "TLSv1.0": ssl.PROTOCOL_TLSv1,
            "TLSv1.1": ssl.PROTOCOL_TLSv1_1,
            "TLSv1.2": ssl.PROTOCOL_TLSv1_2,
            "TLSv1.3": getattr(ssl, 'PROTOCOL_TLSv1_3', None)
        }
        
        support = {}
        
        for protocol_name, protocol_const in protocols.items():
            if protocol_const is None:
                support[protocol_name] = False
                continue
                
            try:
                context = ssl.SSLContext(protocol_const)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                with socket.create_connection((hostname, port), timeout=5) as sock:
                    with context.wrap_socket(sock) as ssock:
                        support[protocol_name] = True
                        
            except Exception:
                support[protocol_name] = False
        
        return support
    
    async def _check_cipher_suites(self, hostname: str, port: int) -> List[str]:
        """Check supported cipher suites."""
        cipher_suites = []
        
        try:
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cipher = ssock.cipher()
                    if cipher:
                        cipher_suites.append({
                            "name": cipher[0],
                            "protocol": cipher[1],
                            "bits": cipher[2]
                        })
                        
        except Exception as e:
            logger.error(f"Failed to check cipher suites: {e}")
        
        return cipher_suites
    
    def _identify_vulnerabilities(self, cert_info: Dict, protocol_support: Dict, cipher_suites: List) -> List[Dict[str, Any]]:
        """Identify SSL/TLS vulnerabilities."""
        vulnerabilities = []
        
        # Check for weak protocols
        for protocol in self.weak_protocols:
            if protocol_support.get(protocol, False):
                vulnerabilities.append({
                    "type": "weak_protocol",
                    "description": f"Weak protocol {protocol} is supported",
                    "severity": "medium" if protocol in ["TLSv1.0", "TLSv1.1"] else "high"
                })
        
        # Check certificate issues
        if cert_info.get("is_expired"):
            vulnerabilities.append({
                "type": "expired_certificate",
                "description": "SSL certificate has expired",
                "severity": "critical"
            })
        elif cert_info.get("expires_soon"):
            vulnerabilities.append({
                "type": "expiring_certificate",
                "description": f"SSL certificate expires in {cert_info.get('days_until_expiry')} days",
                "severity": "medium"
            })
        
        # Check for weak cipher suites
        for cipher in cipher_suites:
            cipher_name = cipher.get("name", "")
            for weak_cipher in self.weak_ciphers:
                if weak_cipher in cipher_name:
                    vulnerabilities.append({
                        "type": "weak_cipher",
                        "description": f"Weak cipher suite {cipher_name} is supported",
                        "severity": "medium"
                    })
        
        return vulnerabilities
    
    def _calculate_security_score(self, cert_info: Dict, protocol_support: Dict, cipher_suites: List, vulnerabilities: List) -> int:
        """Calculate overall security score (0-100)."""
        score = 100
        
        # Deduct points for vulnerabilities
        for vuln in vulnerabilities:
            if vuln["severity"] == "critical":
                score -= 30
            elif vuln["severity"] == "high":
                score -= 20
            elif vuln["severity"] == "medium":
                score -= 10
            elif vuln["severity"] == "low":
                score -= 5
        
        # Bonus points for strong configuration
        if protocol_support.get("TLSv1.3", False):
            score += 5
        
        if not any(protocol_support.get(p, False) for p in self.weak_protocols):
            score += 10
        
        return max(0, min(100, score))


class NetworkSecurityManager:
    """Main network security management class."""
    
    def __init__(self):
        self.ids = IntrusionDetectionSystem()
        self.firewall = NetworkFirewall()
        self.ssl_checker = SSLSecurityChecker()
        self.threat_intel = ThreatIntelligence()
        
        # Start background tasks
        asyncio.create_task(self._background_tasks())
    
    async def _background_tasks(self) -> None:
        """Run background security tasks."""
        while True:
            try:
                # Update threat intelligence every hour
                await self.threat_intel.update_threat_feeds()
                
                # Clean up old connection tracking data
                await self._cleanup_old_data()
                
                # Wait 1 hour before next update
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Background task error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old tracking data to prevent memory leaks."""
        cutoff_time = time.time() - 3600  # 1 hour ago
        
        for ip in list(self.ids.connection_tracking.keys()):
            # Remove old entries
            while (self.ids.connection_tracking[ip] and 
                   self.ids.connection_tracking[ip][0] < cutoff_time):
                self.ids.connection_tracking[ip].popleft()
            
            # Remove empty deques
            if not self.ids.connection_tracking[ip]:
                del self.ids.connection_tracking[ip]
    
    async def analyze_request(self, request: Request) -> SecurityEvent:
        """Analyze incoming request for security threats."""
        return await self.ids.analyze_request(request)
    
    def check_firewall(self, source_ip: str, dest_ip: str, port: int, protocol: str = "tcp") -> str:
        """Check firewall rules for connection."""
        return self.firewall.check_connection(source_ip, dest_ip, port, protocol)
    
    async def check_ssl_security(self, hostname: str, port: int = 443) -> Dict[str, Any]:
        """Check SSL/TLS security configuration."""
        return await self.ssl_checker.check_ssl_configuration(hostname, port)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall network security status."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "blocked_ips": len(self.ids.blocked_ips),
            "firewall_rules": len(self.firewall.rules),
            "threat_intel_last_update": (
                self.threat_intel.last_update.isoformat() 
                if self.threat_intel.last_update else None
            ),
            "malicious_ips_count": len(self.threat_intel.malicious_ips),
            "tor_nodes_count": len(self.threat_intel.tor_exit_nodes),
            "active_connections": len(self.ids.connection_tracking)
        }


# Global network security manager instance
network_security = NetworkSecurityManager()