#!/usr/bin/env python3
"""Setup script for Mem0AI database tables in Supabase."""

import os
import sys
from pathlib import Path

# Load environment variables from .env.orbstack
env_file = Path(__file__).parent / ".env.orbstack"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

try:
    import psycopg2
except ImportError:
    print("Error: psycopg2 not installed. Install with: uv add psycopg2-binary")
    sys.exit(1)

def setup_database():
    """Set up the database with all required tables and extensions."""
    
    # Database connection
    database_url = os.environ.get('MEMORY_DATABASE_URL')
    if not database_url:
        print("Error: MEMORY_DATABASE_URL not found in environment")
        return False
    
    print(f"Connecting to database...")
    
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("✅ Connection successful!")
        
        # Check PostgreSQL version
        cursor.execute('SELECT version();')
        version = cursor.fetchone()[0]
        print(f"PostgreSQL version: {version.split(',')[0]}")
        
        # Step 1: Enable pgvector extension
        print("\n🔧 Setting up pgvector extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";")
        print("✅ Extensions enabled")
        
        # Step 2: Load and execute setup SQL files
        sql_files = [
            "docs/sql/setup_pgvector.sql",
            "docs/sql/supabase_schema.sql"
        ]
        
        for sql_file in sql_files:
            file_path = Path(__file__).parent / sql_file
            if file_path.exists():
                print(f"\n📄 Executing {sql_file}...")
                with open(file_path, 'r') as f:
                    sql_content = f.read()
                
                # Split into individual statements and execute
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                
                executed = 0
                skipped = 0
                
                for i, statement in enumerate(statements):
                    if not statement or statement.startswith('--'):
                        continue
                    
                    try:
                        cursor.execute(statement)
                        executed += 1
                    except psycopg2.Error as e:
                        if "already exists" in str(e).lower():
                            skipped += 1
                        else:
                            print(f"⚠️  Warning in statement {i+1}: {e}")
                
                print(f"✅ Executed {executed} statements, skipped {skipped} existing objects")
            else:
                print(f"⚠️  SQL file not found: {sql_file}")
        
        # Step 3: Verify setup
        print("\n🔍 Verifying setup...")
        
        # Check for key tables
        tables_to_check = [
            'public.memories',
            'public.profiles', 
            'public.memory_sessions',
            'mem0_vectors.memories'
        ]
        
        for table in tables_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema || '.' || table_name = %s;", (table,))
            exists = cursor.fetchone()[0] > 0
            status = "✅" if exists else "❌"
            print(f"{status} Table {table}: {'exists' if exists else 'missing'}")
        
        # Check indexes
        cursor.execute("""
            SELECT COUNT(*) FROM pg_indexes 
            WHERE tablename IN ('memories') 
            AND indexname LIKE '%embedding%'
        """)
        vector_indexes = cursor.fetchone()[0]
        print(f"✅ Vector indexes: {vector_indexes} found")
        
        # Check functions
        cursor.execute("""
            SELECT COUNT(*) FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = 'public' 
            AND p.proname LIKE '%search%memory%'
        """)
        search_functions = cursor.fetchone()[0]
        print(f"✅ Search functions: {search_functions} found")
        
        cursor.close()
        conn.close()
        
        print(f"\n🎉 Database setup completed successfully!")
        return True
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)