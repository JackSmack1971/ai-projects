import json
import os
import time
import threading
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from functools import lru_cache

# Import centralized logging configuration
from logging_config import logger

# Import Config
from config import config

# Import agent initialization functions and TOOL_REGISTRY (will be passed as arguments)
# from agent_factory import initialize_agents, initialize_agent_nodes # Not imported directly, passed as args
# from main import TOOL_REGISTRY # Not imported directly, passed as arg

# Define AgentRole dataclass
@dataclass
class AgentRole:
    name: str
    system_prompt: str
    tools: List[Any]
    dependencies: List[str] = field(default_factory=list)

# Role Registry
ROLE_REGISTRY: Dict[str, AgentRole] = {}

def register_role(role: AgentRole) -> None:
    """Register an agent role."""
    ROLE_REGISTRY[role.name.lower()] = role
    logger.info("Registered agent role: role_name=%s", role.name)

# Keep track of the last modified time for hot-reloading
_roles_config_last_modified: Optional[float] = None

def load_roles_from_config(
    config_path: str,
    tool_registry: Dict[str, Callable],
    init_agents_func: Callable,
    init_agent_nodes_func: Callable,
    get_api_key_func: Callable,
    force_reload: bool = False
) -> None:
    """
    Load agent roles from a JSON configuration file and populate the registry.
    Supports hot-reloading if the file has changed.

    Args:
        config_path (str): The path to the roles JSON configuration file.
        tool_registry (Dict[str, Callable]): The registry of available tools.
        init_agents_func (Callable): Function to re-initialize agents.
        init_agent_nodes_func (Callable): Function to re-initialize agent nodes.
        get_api_key_func (Callable): Function to get the API key.
        force_reload (bool): If True, forces a reload even if the file hasn't changed.
    """
    global _roles_config_last_modified

    try:
        # Validate and sanitize config_path
        # Assuming roles.json should be in the current workspace directory
        base_dir = os.path.abspath(".")
        abs_config_path = os.path.abspath(config_path)
        normalized_config_path = os.path.normpath(abs_config_path)

        # Check if the normalized path is within the base directory
        if not normalized_config_path.startswith(base_dir):
            logger.error("Attempted to load roles config from outside allowed directory: path=%s", config_path)
            # Fallback to default roles or handle as critical error
            logger.info("Using default hardcoded roles due to invalid config file path.")
            ROLE_REGISTRY.clear() # Clear before adding defaults
            register_role(AgentRole(name="Pilot", system_prompt="You are a fully qualified pilot. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
            register_role(AgentRole(name="Copilot", system_prompt="You are a fully qualified copilot. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
            register_role(AgentRole(name="CSO", system_prompt="You are a fully qualified Combat Systems Operator. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
            _roles_config_last_modified = None # Reset modified time on fallback
            return # Exit function after handling invalid path


        current_modified_time = os.path.getmtime(config_path)
        if not force_reload and _roles_config_last_modified is not None and current_modified_time <= _roles_config_last_modified:
            logger.debug("Roles configuration file has not changed since last load.")
            return # No changes, no reload needed

        logger.info("Loading agent roles from configuration file: path=%s", config_path)
        with open(config_path, 'r') as f:
            roles_data = json.load(f)

        # Clear existing roles before loading new ones for hot-reloading
        ROLE_REGISTRY.clear()
        logger.info("Cleared existing role registry for hot-reloading.")

        for role_data in roles_data:
            role_name = role_data.get("name")
            system_prompt = role_data.get("system_prompt")
            tool_names = role_data.get("tools", [])
            dependencies = role_data.get("dependencies", [])

            if not role_name or not system_prompt:
                logger.warning("Skipping role with missing name or system_prompt: role_data=%s", role_data)
                continue

            # Resolve tool names to tool functions
            tools = []
            for tool_name in tool_names:
                tool_func = tool_registry.get(tool_name)
                if tool_func:
                    tools.append(tool_func)
                else:
                   logger.warning("Tool not found in registry: tool_name=%s, role_name=%s", tool_name, role_name)

            register_role(AgentRole(name=role_name, system_prompt=system_prompt, tools=tools, dependencies=dependencies))

        _roles_config_last_modified = current_modified_time
        logger.info("Successfully loaded agent roles: num_roles=%s", len(ROLE_REGISTRY))

        # Re-initialize agents and nodes after roles are reloaded
        # Note: This is a simple approach. A more robust system would handle
        # ongoing missions gracefully during a hot-swap.
        api_key = get_api_key_func() # Re-get API key
        init_agents_func(api_key)
        init_agent_nodes_func()
        logger.info("Re-initialized agents and nodes after role reload.")


    except FileNotFoundError:
        logger.error("Roles configuration file not found: path=%s", config_path)
        # Fallback to default roles or handle as critical error
        logger.info("Using default hardcoded roles due to config file not found.")
        # Define and register default roles if config fails
        ROLE_REGISTRY.clear() # Clear before adding defaults
        register_role(AgentRole(name="Pilot", system_prompt="You are a fully qualified pilot. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
        register_role(AgentRole(name="Copilot", system_prompt="You are a fully qualified copilot. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
        register_role(AgentRole(name="CSO", system_prompt="You are a fully qualified Combat Systems Operator. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
        _roles_config_last_modified = None # Reset modified time on fallback

    except json.JSONDecodeError:
        logger.error("Error decoding roles JSON configuration file: path=%s", config_path)
        # Fallback to default roles or handle as critical error
        logger.info("Using default hardcoded roles due to JSON decoding error.")
        ROLE_REGISTRY.clear() # Clear before adding defaults
        register_role(AgentRole(name="Pilot", system_prompt="You are a fully qualified pilot. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
        register_role(AgentRole(name="Copilot", system_prompt="You are a fully qualified copilot. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
        register_role(AgentRole(name="CSO", system_prompt="You are a fully qualified Combat Systems Operator. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
        _roles_config_last_modified = None # Reset modified time on fallback

    except Exception as e:
        logger.error("An unexpected error occurred while loading roles configuration: path=%s, error=%s", config_path, str(e))
        # Fallback to default roles or handle as critical error
        logger.info("Using default hardcoded roles due to unexpected error.")
        ROLE_REGISTRY.clear() # Clear before adding defaults
        register_role(AgentRole(name="Pilot", system_prompt="You are a fully qualified pilot. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
        register_role(AgentRole(name="Copilot", system_prompt="You are a fully qualified copilot. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
        register_role(AgentRole(name="CSO", system_prompt="You are a fully qualified Combat Systems Operator. Speak only as your role.", tools=[tool_registry.get("retrieve_information")], dependencies=[]))
        _roles_config_last_modified = None # Reset modified time on fallback

# Periodically checks for changes in the roles config file and reloads if necessary.
def role_reloader_thread(
    config_path: str,
    interval_sec: int,
    tool_registry: Dict[str, Callable],
    init_agents_func: Callable,
    init_agent_nodes_func: Callable,
    get_api_key_func: Callable
):
    """Periodically checks for changes in the roles config file and reloads if necessary."""
    logger.info("Starting role reloader thread: config_path=%s, interval_sec=%s", config_path, interval_sec)
    while True:
        time.sleep(interval_sec)
        try:
            load_roles_from_config(
                config_path,
                tool_registry,
                init_agents_func,
                init_agent_nodes_func,
                get_api_key_func
            )
        except Exception as e:
            logger.error("Error during role reloading: error=%s", str(e))

def get_role(role_name: str) -> Optional[AgentRole]:
    """Retrieve a role from the registry."""
    return ROLE_REGISTRY.get(role_name.lower())