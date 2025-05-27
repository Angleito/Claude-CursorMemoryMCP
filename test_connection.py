#!/usr/bin/env python3
"""Test database connection and provide setup instructions."""

import os
import sys
from pathlib import Path

def test_connection():
    """Test database connection with different methods."""
    
    print("🔍 Testing database connection...")
    
    # Load environment
    env_file = Path(__file__).parent / ".env.orbstack"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    database_url = os.environ.get('MEMORY_DATABASE_URL', '')
    
    if not database_url:
        print("❌ MEMORY_DATABASE_URL not found")
        return False
    
    print(f"Database URL: {database_url[:50]}...")
    
    # Parse URL components
    if database_url.startswith('postgresql://'):
        parts = database_url.replace('postgresql://', '').split('@')
        if len(parts) == 2:
            auth_part = parts[0]
            host_part = parts[1]
            
            if ':' in auth_part:
                username, password = auth_part.split(':', 1)
                print(f"Username: {username}")
                print(f"Password: {'*' * len(password)}")
            
            if '/' in host_part:
                host_db = host_part.split('/')
                host_port = host_db[0]
                database = host_db[1] if len(host_db) > 1 else 'postgres'
                
                if ':' in host_port:
                    host, port = host_port.split(':', 1)
                    print(f"Host: {host}")
                    print(f"Port: {port}")
                    print(f"Database: {database}")
    
    # Try to install and test psycopg2
    try:
        import psycopg2
        print("✅ psycopg2 is available")
        
        try:
            print("🔗 Testing connection...")
            conn = psycopg2.connect(database_url, connect_timeout=10)
            print("✅ Connection successful!")
            
            cursor = conn.cursor()
            cursor.execute('SELECT version();')
            version = cursor.fetchone()[0]
            print(f"PostgreSQL: {version.split(',')[0]}")
            
            # Test pgvector
            cursor.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';")
            has_vector = cursor.fetchone()[0] > 0
            print(f"pgvector extension: {'✅ installed' if has_vector else '❌ not installed'}")
            
            cursor.close()
            conn.close()
            return True
            
        except psycopg2.OperationalError as e:
            print(f"❌ Connection failed: {e}")
            print("\n🔧 Troubleshooting steps:")
            print("1. Verify your Supabase project is active")
            print("2. Check the database URL in your Supabase dashboard:")
            print("   - Go to Settings → Database → Connection String")
            print("   - Copy the 'URI' connection string")
            print("3. Ensure your IP is allowed (if using IP restrictions)")
            print("4. Try connecting from Supabase SQL Editor first")
            return False
            
    except ImportError:
        print("❌ psycopg2 not available")
        print("Run: uv add psycopg2-binary")
        return False

def manual_setup_instructions():
    """Provide manual setup instructions."""
    print("\n📋 Manual Database Setup Instructions:")
    print("=" * 50)
    
    print("\n1. 🌐 Go to your Supabase project dashboard")
    print("2. 📊 Navigate to 'SQL Editor'")
    print("3. 📝 Run these commands one by one:")
    
    commands = [
        "-- Enable extensions",
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";", 
        "CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";",
        "",
        "-- Create basic memory table",
        """CREATE TABLE IF NOT EXISTS public.memories (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);""",
        "",
        "-- Create index for vector search",
        """CREATE INDEX IF NOT EXISTS memories_embedding_idx 
ON public.memories USING hnsw (embedding vector_cosine_ops);""",
        "",
        "-- Create user profiles table",
        """CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    display_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);"""
    ]
    
    for cmd in commands:
        print(cmd)
    
    print(f"\n4. ✅ Verify setup by running:")
    print("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';")
    print("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'memories';")

if __name__ == "__main__":
    success = test_connection()
    
    if not success:
        manual_setup_instructions()
        
    print(f"\n{'✅ Setup complete!' if success else '📋 Follow manual setup instructions above'}")