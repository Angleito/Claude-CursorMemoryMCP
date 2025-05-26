#!/usr/bin/env python3
"""Setup script for Mem0 AI MCP Server
Handles installation, configuration, and initial setup.
"""

import json
import secrets
import subprocess
import sys
from pathlib import Path


class Mem0Setup:
    """Setup manager for Mem0 AI MCP Server."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"
        self.plugins_dir = self.project_root / "plugins"

    def run_setup(self):
        """Run complete setup process."""
        print("🚀 Mem0 AI MCP Server Setup")
        print("=" * 40)

        try:
            self.check_python_version()
            self.install_dependencies()
            self.create_directories()
            self.setup_environment()
            self.setup_database()
            self.configure_mcp_clients()
            self.run_tests()

            print("\n✅ Setup completed successfully!")
            print("\nNext steps:")
            print("1. Configure your .env file with your API keys")
            print("2. Start the server: python main.py")
            print("3. Test MCP integration with Claude Code or Cursor")

        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            sys.exit(1)

    def check_python_version(self):
        """Check Python version compatibility."""
        print("📋 Checking Python version...")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            raise RuntimeError("Python 3.8+ is required. Python 3.12+ is strongly recommended for optimal performance and all features.")

        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        
        if version.major == 3 and version.minor < 12:
            print(f"⚠️  Consider upgrading to Python 3.12+ for better performance and latest features")

    def install_uv(self):
        """Install uv if not already available."""
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
            print("✓ uv is already installed")
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
            
        print("📥 Installing uv...")
        try:
            # Install uv using the official installer
            install_script = subprocess.run(
                ["curl", "-LsSf", "https://astral.sh/uv/install.sh"],
                capture_output=True,
                text=True,
                check=True
            )
            subprocess.run(
                ["sh"],
                input=install_script.stdout,
                text=True,
                check=True
            )
            
            # Verify installation
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
            print("✓ uv installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install uv automatically: {e}")
            print("Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/")
            print("Continuing with pip as fallback...")

    def install_dependencies(self):
        """Install Python dependencies using uv with pip fallback."""
        print("\n📦 Installing dependencies...")

        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            raise FileNotFoundError("requirements.txt not found")

        # First ensure uv is installed
        self.install_uv()

        try:
            # Try uv first - create venv and install dependencies
            print("🚀 Using uv for dependency management...")
            
            # Create virtual environment with uv
            try:
                subprocess.run(
                    ["uv", "venv", ".venv"],
                    cwd=self.project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("✓ Virtual environment created with uv")
            except subprocess.CalledProcessError:
                print("⚠️  Virtual environment may already exist")
            
            # Install dependencies with uv
            subprocess.run(
                ["uv", "pip", "install", "-r", str(requirements_file)],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            print("✓ Dependencies installed with uv")
            
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"⚠️  uv installation failed: {e}")
            print("🔄 Falling back to pip...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("✓ Dependencies installed with pip (fallback)")
            except subprocess.CalledProcessError as pip_error:
                print(f"❌ Failed to install dependencies: {pip_error.stderr}")
                raise

    def create_directories(self):
        """Create necessary directories."""
        print("\n📁 Creating directories...")

        directories = [
            self.logs_dir,
            self.plugins_dir,
            self.project_root / "ssl",
            self.project_root / "backups",
        ]

        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"✓ Created {directory.name}/")

    def setup_environment(self):
        """Setup environment configuration."""
        print("\n⚙️  Setting up environment...")

        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"

        if env_file.exists():
            print("✓ .env file already exists")
            return

        if env_example.exists():
            # Copy example and generate secrets
            with open(env_example) as f:
                content = f.read()

            # Generate secret key
            secret_key = secrets.token_urlsafe(32)
            content = content.replace("your_secret_key_here", secret_key)

            with open(env_file, "w") as f:
                f.write(content)

            print("✓ Created .env file from template")
            print("⚠️  Please update .env with your API keys and database credentials")
        else:
            print("⚠️  .env.example not found, please create .env manually")

    def setup_database(self):
        """Setup database if needed."""
        print("\n🗄️  Setting up database...")

        # Create database initialization script
        init_sql = self.project_root / "scripts" / "init.sql"
        init_sql.parent.mkdir(exist_ok=True)

        sql_content = """
-- Mem0 AI Database Initialization Script
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE mem0ai' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mem0ai');

-- Use the mem0ai database
\\c mem0ai;

-- Enable extensions in the mem0ai database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create memories table with vector support
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    memory_type VARCHAR(50) DEFAULT 'fact',
    priority VARCHAR(20) DEFAULT 'medium',
    source VARCHAR(255),
    context TEXT,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_memories_metadata ON memories USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);

-- Create search history table
CREATE TABLE IF NOT EXISTS search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    results_count INTEGER,
    execution_time_ms FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create plugin configurations table
CREATE TABLE IF NOT EXISTS plugin_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plugin_name VARCHAR(255) NOT NULL,
    config JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, plugin_name)
);

-- Create trigger function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_memories_updated_at ON memories;
CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_plugin_configs_updated_at ON plugin_configs;
CREATE TRIGGER update_plugin_configs_updated_at
    BEFORE UPDATE ON plugin_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create a default user (optional)
-- INSERT INTO users (username, email, password_hash, full_name)
-- VALUES ('admin', 'admin@mem0ai.com', '$2b$12$placeholder_hash', 'Administrator')
-- ON CONFLICT (username) DO NOTHING;

COMMIT;
"""

        with open(init_sql, "w") as f:
            f.write(sql_content)

        print("✓ Database initialization script created")

    def configure_mcp_clients(self):
        """Configure MCP clients."""
        print("\n🔧 Configuring MCP clients...")

        # Update paths in configuration files
        configs = [
            self.config_dir / "claude-code-mcp.json",
            self.config_dir / "cursor-mcp.json",
        ]

        for config_file in configs:
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

                # Update paths to absolute paths
                if "mcpServers" in config:
                    for _server_name, server_config in config["mcpServers"].items():
                        if "args" in server_config:
                            # Update main.py path to absolute path
                            for i, arg in enumerate(server_config["args"]):
                                if arg.endswith("main.py"):
                                    server_config["args"][i] = str(
                                        self.project_root / "main.py"
                                    )

                        if (
                            "env" in server_config
                            and "PYTHONPATH" in server_config["env"]
                        ):
                            server_config["env"]["PYTHONPATH"] = str(self.project_root)

                with open(config_file, "w") as f:
                    json.dump(config, f, indent=2)

                print(f"✓ Updated {config_file.name}")

    def run_tests(self):
        """Run basic tests to verify setup."""
        print("\n🧪 Running basic tests...")

        try:
            # Test imports
            import fastapi
            import openai
            import redis
            import supabase
            import uvicorn

            print("✓ All dependencies importable")

            # Test configuration loading
            sys.path.insert(0, str(self.project_root))
            try:
                from src.config import Settings

                # This will fail if required env vars are missing, which is expected
                print("✓ Configuration module loadable")
            except Exception:
                print("⚠️  Configuration validation requires .env setup")

        except ImportError as e:
            print(f"❌ Import test failed: {e}")
            raise


def main():
    """Main setup function."""
    setup = Mem0Setup()
    setup.run_setup()


if __name__ == "__main__":
    main()
