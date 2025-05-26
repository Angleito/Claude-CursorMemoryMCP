"""Plugin architecture for extensibility.

This module provides a comprehensive plugin system allowing users to extend
the Mem0 AI MCP Server with custom functionality. It supports memory plugins,
search enhancement plugins, and notification plugins with automatic loading
and configuration management.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import sys
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

import structlog

if TYPE_CHECKING:
    from .config import Settings
from .models import PluginConfig

logger = structlog.get_logger()


class PluginBase(ABC):
    """Base class for all plugins.

    All plugins must inherit from this class and implement the required methods.
    The plugin system provides configuration management, lifecycle control,
    and schema validation.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.name = self.__class__.__name__
        self.version = getattr(self.__class__, "VERSION", "1.0.0")
        self.description = getattr(self.__class__, "DESCRIPTION", "")
        self.enabled = True

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin."""

    @abstractmethod
    async def execute(
        self, data: dict[str, Any], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute the plugin with given data."""

    async def cleanup(self) -> None:
        """Cleanup plugin resources.

        Override this method to perform any necessary cleanup when the plugin
        is being unloaded or the system is shutting down.
        """

    def get_schema(self) -> dict[str, Any]:
        """Get the plugin's input/output schema."""
        return {"input": {"type": "object"}, "output": {"type": "object"}}

    def get_info(self) -> dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enabled": self.enabled,
            "schema": self.get_schema(),
        }


class MemoryPlugin(PluginBase):
    """Base class for memory-related plugins.

    Memory plugins can process memory content during creation, update,
    or retrieval operations. They have access to the memory manager
    for advanced operations.
    """

    def __init__(self, memory_manager: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.memory_manager = memory_manager


class SearchPlugin(PluginBase):
    """Base class for search enhancement plugins.

    Search plugins can enhance search queries and filter results to provide
    better search experiences and more relevant results.
    """

    async def enhance_query(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Enhance search query.

        Args:
            query: Original search query
            context: Optional context information

        Returns:
            Enhanced query string
        """
        return query

    async def filter_results(
        self, results: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Filter search results.

        Args:
            results: List of search results
            context: Optional context information

        Returns:
            Filtered list of results
        """
        return results


class NotificationPlugin(PluginBase):
    """Base class for notification plugins.

    Notification plugins can send notifications via various channels
    such as email, SMS, webhooks, or push notifications.
    """

    async def send_notification(
        self, message: str, recipient: str, data: dict[str, Any] | None = None
    ) -> None:
        """Send notification.

        Args:
            message: Notification message
            recipient: Recipient identifier (email, phone, etc.)
            data: Optional additional data for the notification
        """


class PluginManager:
    """Manages plugin loading, execution, and lifecycle.

    This class handles:
    - Dynamic plugin discovery and loading
    - Plugin configuration management
    - Plugin execution with error handling
    - Plugin lifecycle management (enable/disable/reload)
    - Example plugin generation
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.plugins: dict[str, PluginBase] = {}
        self.plugin_configs: dict[str, PluginConfig] = {}
        self.enabled = settings.enable_plugins

    async def load_plugins(self) -> None:
        """Load all plugins from the plugins directory.

        This method discovers and loads all Python files in the plugins directory,
        instantiates plugin classes, and configures them.
        """
        if not self.enabled:
            logger.info("Plugins disabled in configuration")
            return

        plugins_dir = self.settings.plugins_dir
        if not os.path.exists(plugins_dir):
            logger.info("Plugins directory not found, creating", path=plugins_dir)
            os.makedirs(plugins_dir, exist_ok=True)
            await self._create_example_plugins(plugins_dir)
            return

        if plugins_dir not in sys.path:
            sys.path.insert(0, plugins_dir)

        try:
            for filename in os.listdir(plugins_dir):
                if filename.endswith(".py") and not filename.startswith("_"):
                    module_name = filename[:-3]
                    await self._load_plugin_module(module_name, plugins_dir)

            logger.info("Plugins loaded successfully", count=len(self.plugins))

        except Exception as e:
            logger.error("Failed to load plugins", error=str(e))

    async def _load_plugin_module(self, module_name: str, plugins_dir: str):
        """Load a specific plugin module."""
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(plugins_dir, f"{module_name}.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes in the module
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, PluginBase)
                    and obj != PluginBase
                    and not inspect.isabstract(obj)
                ):

                    # Load plugin configuration
                    plugin_config = await self._load_plugin_config(name)

                    # Initialize plugin
                    if issubclass(obj, MemoryPlugin):
                        # Inject memory manager for memory plugins
                        plugin_instance = obj(
                            None, plugin_config.settings
                        )  # Memory manager injected later
                    else:
                        plugin_instance = obj(plugin_config.settings)

                    # Initialize the plugin
                    if await plugin_instance.initialize():
                        self.plugins[name] = plugin_instance
                        self.plugin_configs[name] = plugin_config
                        logger.info(
                            "Plugin loaded", name=name, version=plugin_instance.version
                        )
                    else:
                        logger.warning("Plugin initialization failed", name=name)

        except Exception as e:
            logger.error(
                "Failed to load plugin module", module=module_name, error=str(e)
            )

    async def _load_plugin_config(self, plugin_name: str) -> PluginConfig:
        """Load plugin configuration."""
        try:
            config_path = os.path.join(self.settings.plugins_dir, f"{plugin_name}.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config_data = json.load(f)
                return PluginConfig(**config_data)
            else:
                # Create default config
                default_config = PluginConfig(
                    name=plugin_name,
                    version="1.0.0",
                    description=f"Configuration for {plugin_name} plugin",
                    enabled=True,
                    settings={},
                )
                await self._save_plugin_config(plugin_name, default_config)
                return default_config

        except Exception as e:
            logger.error(
                "Failed to load plugin config", plugin=plugin_name, error=str(e)
            )
            return PluginConfig(
                name=plugin_name, version="1.0.0", description="", enabled=True
            )

    async def _save_plugin_config(self, plugin_name: str, config: PluginConfig):
        """Save plugin configuration."""
        try:
            config_path = os.path.join(self.settings.plugins_dir, f"{plugin_name}.json")
            with open(config_path, "w") as f:
                json.dump(config.dict(), f, indent=2)
        except Exception as e:
            logger.error(
                "Failed to save plugin config", plugin=plugin_name, error=str(e)
            )

    async def execute_plugin(
        self, plugin_name: str, data: dict[str, Any], user_id: str
    ) -> dict[str, Any]:
        """Execute a specific plugin."""
        if not self.enabled:
            raise ValueError("Plugins are disabled")

        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")

        plugin = self.plugins[plugin_name]
        config = self.plugin_configs.get(plugin_name)

        if not config or not config.enabled:
            raise ValueError(f"Plugin '{plugin_name}' is disabled")

        try:
            context = {
                "user_id": user_id,
                "plugin_name": plugin_name,
                "config": config.settings,
            }

            result = await plugin.execute(data, context)

            logger.info(
                "Plugin executed successfully", plugin=plugin_name, user_id=user_id
            )
            return result

        except Exception as e:
            logger.error("Plugin execution failed", plugin=plugin_name, error=str(e))
            raise

    async def list_plugins(self) -> list[dict[str, Any]]:
        """List all available plugins."""
        return [plugin.get_info() for plugin in self.plugins.values()]

    async def get_plugin_info(self, plugin_name: str) -> dict[str, Any] | None:
        """Get information about a specific plugin."""
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].get_info()
        return None

    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self.plugin_configs:
            config = self.plugin_configs[plugin_name]
            config.enabled = True
            await self._save_plugin_config(plugin_name, config)

            if plugin_name in self.plugins:
                self.plugins[plugin_name].enabled = True

            return True
        return False

    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.plugin_configs:
            config = self.plugin_configs[plugin_name]
            config.enabled = False
            await self._save_plugin_config(plugin_name, config)

            if plugin_name in self.plugins:
                self.plugins[plugin_name].enabled = False

            return True
        return False

    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin."""
        try:
            # Cleanup existing plugin
            if plugin_name in self.plugins:
                await self.plugins[plugin_name].cleanup()
                del self.plugins[plugin_name]

            if plugin_name in self.plugin_configs:
                del self.plugin_configs[plugin_name]

            # Reload the plugin
            await self._load_plugin_module(plugin_name, self.settings.plugins_dir)

            return plugin_name in self.plugins

        except Exception as e:
            logger.error("Failed to reload plugin", plugin=plugin_name, error=str(e))
            return False

    async def _create_example_plugins(self, plugins_dir: str):
        """Create example plugins for demonstration."""
        # Example memory summarizer plugin
        summarizer_plugin = '''"""Example memory summarizer plugin"""

import asyncio
from typing import Dict, Any
from src.plugins import MemoryPlugin


class MemorySummarizer(MemoryPlugin):
    """Plugin to summarize long memories"""

    VERSION = "1.0.0"
    DESCRIPTION = "Summarizes long memory content to key points"

    async def initialize(self) -> bool:
        self.max_length = self.config.get("max_length", 200)
        return True

    async def execute(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        content = data.get("content", "")

        if len(content) <= self.max_length:
            return {"summary": content, "summarized": False}

        # Simple summarization (in real implementation, use AI)
        sentences = content.split(". ")
        key_sentences = sentences[:3]  # Take first 3 sentences
        summary = ". ".join(key_sentences)

        if len(summary) > self.max_length:
            summary = summary[:self.max_length] + "..."

        return {
            "summary": summary,
            "summarized": True,
            "original_length": len(content),
            "summary_length": len(summary)
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "input": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to summarize"}
                },
                "required": ["content"]
            },
            "output": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "summarized": {"type": "boolean"},
                    "original_length": {"type": "integer"},
                    "summary_length": {"type": "integer"}
                }
            }
        }
'''

        # Example notification plugin
        notification_plugin = '''"""Example notification plugin"""

import asyncio
from typing import Dict, Any
from src.plugins import NotificationPlugin


class EmailNotifier(NotificationPlugin):
    """Plugin to send email notifications"""

    VERSION = "1.0.0"
    DESCRIPTION = "Sends email notifications for memory events"

    async def initialize(self) -> bool:
        self.smtp_server = self.config.get("smtp_server", "localhost")
        self.smtp_port = self.config.get("smtp_port", 587)
        self.from_email = self.config.get("from_email", "noreply@mem0ai.com")
        return True

    async def execute(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        message = data.get("message", "")
        recipient = data.get("recipient", "")
        subject = data.get("subject", "Memory Notification")

        # Simulate sending email (implement actual email sending here)
        await asyncio.sleep(0.1)  # Simulate network delay

        return {
            "sent": True,
            "recipient": recipient,
            "subject": subject,
            "message_length": len(message),
            "timestamp": context.get("timestamp") if context else None
        }

    async def send_notification(self, message: str, recipient: str, data: Dict[str, Any] = None):
        return await self.execute({
            "message": message,
            "recipient": recipient,
            "subject": data.get("subject", "Memory Notification") if data else "Memory Notification"
        })

    def get_schema(self) -> Dict[str, Any]:
        return {
            "input": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Notification message"},
                    "recipient": {"type": "string", "description": "Email recipient"},
                    "subject": {"type": "string", "description": "Email subject"}
                },
                "required": ["message", "recipient"]
            },
            "output": {
                "type": "object",
                "properties": {
                    "sent": {"type": "boolean"},
                    "recipient": {"type": "string"},
                    "subject": {"type": "string"}
                }
            }
        }
'''

        # Write example plugins
        with open(os.path.join(plugins_dir, "memory_summarizer.py"), "w") as f:
            f.write(summarizer_plugin)

        with open(os.path.join(plugins_dir, "email_notifier.py"), "w") as f:
            f.write(notification_plugin)

        # Create example configurations
        summarizer_config = {
            "name": "MemorySummarizer",
            "version": "1.0.0",
            "description": "Summarizes long memory content",
            "enabled": True,
            "settings": {"max_length": 200},
        }

        notifier_config = {
            "name": "EmailNotifier",
            "version": "1.0.0",
            "description": "Sends email notifications",
            "enabled": False,  # Disabled by default
            "settings": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "from_email": "noreply@mem0ai.com",
            },
        }

        with open(os.path.join(plugins_dir, "MemorySummarizer.json"), "w") as f:
            json.dump(summarizer_config, f, indent=2)

        with open(os.path.join(plugins_dir, "EmailNotifier.json"), "w") as f:
            json.dump(notifier_config, f, indent=2)

        logger.info("Example plugins created", plugins_dir=plugins_dir)

    async def close(self) -> None:
        """Close plugin manager and cleanup all plugins.

        This method gracefully shuts down all plugins and cleans up resources.
        """
        for plugin in self.plugins.values():
            try:
                await plugin.cleanup()
            except Exception as e:
                logger.error(
                    "Error cleaning up plugin", plugin=plugin.name, error=str(e)
                )

        self.plugins.clear()
        self.plugin_configs.clear()
        logger.info("Plugin manager closed")
