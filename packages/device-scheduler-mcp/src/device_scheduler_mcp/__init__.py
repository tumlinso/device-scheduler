"""Authenticated MCP and Python client surfaces for Device Scheduler."""

from .auth import (
    AuthConfigurationError,
    StaticTokenStore,
    StaticTokenVerifier,
    TokenGrant,
    principal_from_request,
    read_client_token,
)
from .client import (
    DeviceSchedulerClient,
    DeviceSchedulerClientError,
    DeviceSchedulerProtocolError,
)
from .config import ConfigurationError, ServiceConfig, load_json_or_toml, load_profiles
from .registry import Operation, operation_names, operations
from .server import create_app, create_asgi_app, create_mcp_server, register_operation_tools
from .stdio import create_stdio_mcp

__all__ = [
    "AuthConfigurationError", "ConfigurationError", "DeviceSchedulerClient", "DeviceSchedulerClientError",
    "DeviceSchedulerProtocolError", "Operation", "ServiceConfig", "StaticTokenStore", "StaticTokenVerifier",
    "TokenGrant", "create_app", "create_asgi_app", "create_mcp_server", "create_stdio_mcp",
    "load_json_or_toml", "load_profiles", "operation_names", "operations", "principal_from_request", "read_client_token",
    "register_operation_tools",
]
