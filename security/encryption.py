"""
Data encryption at rest and in transit
"""
from typing import Union, Dict, Any, Optional, List
from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
import base64
import os
import secrets
import json
from datetime import datetime, timedelta
import hashlib
import hmac

from config.settings import get_settings


settings = get_settings()


class EncryptionManager:
    """Main encryption manager for all cryptographic operations"""
    
    def __init__(self):
        self.settings = settings
        self._fernet_key = None
        self._fernet = None
        self._multi_fernet = None
        self._rsa_private_key = None
        self._rsa_public_key = None
    
    @property
    def fernet_key(self) -> bytes:
        """Get or generate Fernet encryption key"""
        if self._fernet_key is None:
            key_material = self.settings.security.encryption_key
            if not key_material:
                key_material = secrets.token_urlsafe(32)
            
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'mem0ai_encryption_salt',  # Static salt for consistency
                iterations=100000,
            )
            self._fernet_key = base64.urlsafe_b64encode(kdf.derive(key_material.encode()))
        
        return self._fernet_key
    
    @property
    def fernet(self) -> Fernet:
        """Get Fernet cipher instance"""
        if self._fernet is None:
            self._fernet = Fernet(self.fernet_key)
        return self._fernet
    
    @property
    def multi_fernet(self) -> MultiFernet:
        """Get MultiFernet for key rotation"""
        if self._multi_fernet is None:
            # Support key rotation by using multiple keys
            keys = [self.fernet_key]
            
            # Add any additional keys for rotation
            additional_keys = os.environ.get('ENCRYPTION_KEYS_ADDITIONAL', '').split(',')
            for key in additional_keys:
                if key.strip():
                    keys.append(base64.urlsafe_b64decode(key.strip()))
            
            self._multi_fernet = MultiFernet([Fernet(key) for key in keys])
        
        return self._multi_fernet
    
    # Symmetric encryption for data at rest
    def encrypt_data(self, data: Union[str, bytes, dict]) -> str:
        """Encrypt data for storage"""
        if isinstance(data, dict):
            data = json.dumps(data)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt stored data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.multi_fernet.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to decrypt data: {str(e)}")
    
    def encrypt_json(self, data: dict) -> str:
        """Encrypt JSON data"""
        json_str = json.dumps(data, separators=(',', ':'))
        return self.encrypt_data(json_str)
    
    def decrypt_json(self, encrypted_data: str) -> dict:
        """Decrypt JSON data"""
        decrypted_str = self.decrypt_data(encrypted_data)
        return json.loads(decrypted_str)
    
    # Field-level encryption for sensitive database fields
    def encrypt_field(self, field_value: str, field_name: str) -> str:
        """Encrypt a database field with field-specific context"""
        if not field_value:
            return field_value
        
        # Add field context to prevent field substitution attacks
        context = f"{field_name}:{field_value}"
        return self.encrypt_data(context)
    
    def decrypt_field(self, encrypted_value: str, field_name: str) -> str:
        """Decrypt a database field"""
        if not encrypted_value:
            return encrypted_value
        
        try:
            decrypted = self.decrypt_data(encrypted_value)
            
            # Verify field context
            if decrypted.startswith(f"{field_name}:"):
                return decrypted[len(field_name) + 1:]
            else:
                raise ValueError("Field context mismatch")
        except Exception as e:
            raise ValueError(f"Failed to decrypt field {field_name}: {str(e)}")
    
    # Vector embedding encryption
    def encrypt_vector(self, vector: List[float]) -> str:
        """Encrypt vector embeddings"""
        # Convert to bytes for encryption
        vector_bytes = json.dumps(vector).encode('utf-8')
        return self.encrypt_data(vector_bytes)
    
    def decrypt_vector(self, encrypted_vector: str) -> List[float]:
        """Decrypt vector embeddings"""
        decrypted_str = self.decrypt_data(encrypted_vector)
        return json.loads(decrypted_str)
    
    # Memory content encryption
    def encrypt_memory_content(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """Encrypt memory content with metadata"""
        result = {
            'content': self.encrypt_data(content),
            'encrypted_at': datetime.utcnow().isoformat()
        }
        
        if metadata:
            result['metadata'] = self.encrypt_json(metadata)
        
        return result
    
    def decrypt_memory_content(self, encrypted_memory: Dict[str, str]) -> Dict[str, Any]:
        """Decrypt memory content"""
        result = {
            'content': self.decrypt_data(encrypted_memory['content']),
            'encrypted_at': encrypted_memory.get('encrypted_at')
        }
        
        if 'metadata' in encrypted_memory:
            result['metadata'] = self.decrypt_json(encrypted_memory['metadata'])
        
        return result
    
    # Password hashing (one-way encryption)
    def hash_password(self, password: str, salt: bytes = None) -> tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = os.urandom(32)
        
        # Use Scrypt for password hashing (more secure than bcrypt for high-entropy passwords)
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            n=2**14,  # CPU/Memory cost parameter
            r=8,      # Block size parameter
            p=1,      # Parallelization parameter
        )
        
        key = kdf.derive(password.encode('utf-8'))
        
        # Return hash and salt as base64 strings
        password_hash = base64.urlsafe_b64encode(key).decode('utf-8')
        salt_str = base64.urlsafe_b64encode(salt).decode('utf-8')
        
        return password_hash, salt_str
    
    def verify_password(self, password: str, password_hash: str, salt_str: str) -> bool:
        """Verify password against hash"""
        try:
            salt = base64.urlsafe_b64decode(salt_str.encode('utf-8'))
            stored_hash = base64.urlsafe_b64decode(password_hash.encode('utf-8'))
            
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                n=2**14,
                r=8,
                p=1,
            )
            
            kdf.verify(password.encode('utf-8'), stored_hash)
            return True
        except Exception:
            return False
    
    # API key encryption
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key for storage"""
        return self.encrypt_field(api_key, "api_key")
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key"""
        return self.decrypt_field(encrypted_key, "api_key")
    
    # Key rotation support
    def rotate_encryption_key(self) -> str:
        """Generate new encryption key for rotation"""
        new_key = Fernet.generate_key()
        return base64.urlsafe_b64encode(new_key).decode('utf-8')
    
    def migrate_encrypted_data(self, old_encrypted: str, new_fernet: Fernet) -> str:
        """Migrate data from old key to new key"""
        # Decrypt with old key
        decrypted = self.decrypt_data(old_encrypted)
        
        # Encrypt with new key
        encrypted = new_fernet.encrypt(decrypted.encode('utf-8'))
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')


class TLSManager:
    """TLS/SSL configuration manager"""
    
    def __init__(self):
        self.settings = settings
    
    def get_ssl_context(self):
        """Get SSL context for HTTPS"""
        import ssl
        
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Load certificate and key
        if self.settings.server.ssl_cert_path and self.settings.server.ssl_key_path:
            context.load_cert_chain(
                self.settings.server.ssl_cert_path,
                self.settings.server.ssl_key_path
            )
        
        # Security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        return context
    
    def get_client_ssl_context(self):
        """Get SSL context for outbound connections"""
        import ssl
        
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        return context


class DatabaseEncryption:
    """Database-specific encryption utilities"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption = encryption_manager
    
    def encrypt_row(self, row_data: Dict[str, Any], encrypted_fields: List[str]) -> Dict[str, Any]:
        """Encrypt specified fields in a database row"""
        result = row_data.copy()
        
        for field in encrypted_fields:
            if field in result and result[field] is not None:
                result[field] = self.encryption.encrypt_field(str(result[field]), field)
        
        return result
    
    def decrypt_row(self, row_data: Dict[str, Any], encrypted_fields: List[str]) -> Dict[str, Any]:
        """Decrypt specified fields in a database row"""
        result = row_data.copy()
        
        for field in encrypted_fields:
            if field in result and result[field] is not None:
                try:
                    result[field] = self.encryption.decrypt_field(result[field], field)
                except Exception:
                    # Field might not be encrypted (legacy data)
                    pass
        
        return result
    
    def get_encrypted_search_hash(self, value: str, field_name: str) -> str:
        """Get searchable hash for encrypted field"""
        # Create deterministic hash for searching encrypted fields
        combined = f"{field_name}:{value}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()


class TokenEncryption:
    """Token encryption for secure token transmission"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption = encryption_manager
    
    def create_secure_token(self, data: Dict[str, Any], expires_in: int = 3600) -> str:
        """Create encrypted token with expiration"""
        expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        token_data = {
            'data': data,
            'expires_at': expires_at.isoformat(),
            'nonce': secrets.token_urlsafe(16)
        }
        
        return self.encryption.encrypt_json(token_data)
    
    def verify_secure_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decrypt secure token"""
        try:
            token_data = self.encryption.decrypt_json(token)
            
            # Check expiration
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if datetime.utcnow() > expires_at:
                return None
            
            return token_data['data']
        except Exception:
            return None


class BackupEncryption:
    """Encryption for backup data"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption = encryption_manager
    
    def encrypt_backup(self, backup_data: Dict[str, Any]) -> str:
        """Encrypt backup data"""
        backup_with_metadata = {
            'data': backup_data,
            'backup_time': datetime.utcnow().isoformat(),
            'version': '1.0',
            'checksum': self._calculate_checksum(backup_data)
        }
        
        return self.encryption.encrypt_json(backup_with_metadata)
    
    def decrypt_backup(self, encrypted_backup: str) -> Dict[str, Any]:
        """Decrypt and verify backup data"""
        backup_with_metadata = self.encryption.decrypt_json(encrypted_backup)
        
        # Verify checksum
        stored_checksum = backup_with_metadata.get('checksum')
        calculated_checksum = self._calculate_checksum(backup_with_metadata['data'])
        
        if stored_checksum != calculated_checksum:
            raise ValueError("Backup data integrity check failed")
        
        return backup_with_metadata['data']
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity"""
        data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()


# Global encryption manager instance
encryption_manager = EncryptionManager()
tls_manager = TLSManager()
database_encryption = DatabaseEncryption(encryption_manager)
token_encryption = TokenEncryption(encryption_manager)
backup_encryption = BackupEncryption(encryption_manager)