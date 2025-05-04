import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from langgraph.graph import StateGraph, END

# Import centralized logging configuration
from logging_config import logger

# Import Config
import config_loader

# Import AgentRole and ROLE_REGISTRY
from role_manager import AgentRole, ROLE_REGISTRY

# Import AGENT_NODES
from agent_factory import AGENT_NODES

# --- State Management ---
@dataclass
class MissionState:
    """Represents the state of the mission simulation."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())) # Unique ID for each scenario run
    messages: List[str] = field(default_factory=list)
    mission_progress: List[str] = field(default_factory=list)
    next: str = config_loader.Config.DEFAULT_ROLE.lower() # Use string role name
    mission_status: str = "in_progress" # Added mission status field

# --- Agent Output Model ---
class AgentOutput(BaseModel):
    """Expected structure for agent output."""
    reasoning: str
    final_answer: str
    mission_status: Optional[str] = "in_progress" # Agent can suggest mission status

# --- Dynamic Role Assignment (part of graph control flow) ---
def dynamic_role_assignment(state: MissionState) -> str:
    """
    Dynamically assign roles based on the mission progress and role dependencies.

    Args:
        state (MissionState): The current state of the mission.

    Returns:
        str: The assigned role name (lowercase).
    """
    logger.info("Attempting dynamic role assignment: request_id=%s", state.request_id)
    mission_progress = state.mission_progress

    # Basic check for dependencies
    # Iterate through roles and check if their dependencies are met based on recent progress
    # Optimize dependency checking
    recent_progress = state.mission_progress[-5:] # Look at the last 5 progress entries
    # Create a set of role names that have made recent progress
    recent_roles = set()
    for progress in recent_progress:
        # Assuming role name is at the start followed by a colon
        if ":" in progress:
            role_part = progress.split(":", 1)[0].strip().lower()
            recent_roles.add(role_part)

    for role_name, role_config in ROLE_REGISTRY.items():
        if role_config.dependencies:
            dependencies_met = True
            # Check if all dependent roles are in the set of recent roles
            for dep_role in role_config.dependencies:
                if dep_role.lower() not in recent_roles:
                    dependencies_met = False
                    break
            if dependencies_met:
                logger.info("Dependencies met for role, assigning: role_name=%s, dependencies=%s, request_id=%s", role_name, role_config.dependencies, state.request_id)
                return role_name.lower()

    # Fallback to original logic if no dependencies are met for any role
    if any("critical" in progress.lower() for progress in mission_progress):
        logger.info("Assigning pilot role based on critical progress: request_id=%s", state.request_id)
        return "pilot"
    if any("navigation" in progress.lower() for progress in mission_progress):
        logger.info("Assigning cso role based on navigation progress: request_id=%s", state.request_id)
        return "cso"

    logger.info("Assigning default role: default_role=%s, request_id=%s", config_loader.Config.DEFAULT_ROLE.lower(), state.request_id)
    return config_loader.Config.DEFAULT_ROLE.lower()


# --- Graph Nodes ---
def check_mission_status_node(state: MissionState) -> str:
    """
    Checks the current mission status and determines the next step.
    """
    logger.info("Executing check_mission_status_node: request_id=%s, mission_status=%s", state.request_id, state.mission_status)
    if state.mission_status in ["success", "failure", "max_iterations_reached"]:
        logger.info("Mission concluded: request_id=%s, final_status=%s", state.request_id, state.mission_status)
        return state.mission_status # Return the status to the conditional edge
    else:
        logger.info("Mission in progress, returning to command_control: request_id=%s", state.request_id)
        return "in_progress" # Return "in_progress" to the conditional edge

def update_mission_state_node(state: MissionState) -> MissionState:
    """
    Updates the mission state based on agent outputs and determines the next agent.
    This is a simplified example; real-world logic might be more complex.
    """
    start_time = time.perf_counter()
    logger.info("Executing update_mission_state node: request_id=%s", state.request_id)
    # The agent node already updated messages and mission_progress in the state dictionary

    # Determine the next agent based on dynamic role assignment
    next_agent_name = dynamic_role_assignment(state)
    state.next = next_agent_name # Update the state with the next agent

    end_time = time.perf_counter()
    duration = end_time - start_time
    logger.info("Update mission state complete: request_id=%s, next_role=%s, duration_ms=%s", state.request_id, state.next, duration*1000)

    return state # Return the updated state

# --- Graph Construction ---
def create_command_control_graph() -> StateGraph:
    """Creates the LangGraph for command and control."""
    graph = StateGraph(MissionState)

    # Add a node for each agent
    for role_name, node_func in AGENT_NODES.items():
        graph.add_node(role_name, node_func)
        logger.info("Added node to graph: node_name=%s", role_name)

    # Add a node for checking mission status
    graph.add_node("check_mission_status", check_mission_status_node)
    logger.info("Added node to graph: node_name=check_mission_status")

    # Add a node for updating mission state (optional, can be integrated into agent nodes)
    graph.add_node("update_mission_state", update_mission_state_node)
    logger.info("Added node to graph: node_name=update_mission_state")

    # Define the entry point
    graph.set_entry_point(config_loader.Config.DEFAULT_ROLE.lower())
    logger.info("Set entry point: entry_point=%s", config_loader.Config.DEFAULT_ROLE.lower())

    # Define the transitions
    # Each agent node transitions to the check_mission_status node
    for role_name in ROLE_REGISTRY.keys():
        graph.add_edge(role_name, "check_mission_status")
        logger.info("Added edge: source=%s, target=%s", role_name, "check_mission_status")

    # The check_mission_status node transitions based on the mission status
    graph.add_conditional_edges(
        "check_mission_status",
        lambda state: state.mission_status, # The condition is the mission_status field
        {
            "in_progress": "update_mission_state", # If in progress, go to update state
            "success": END, # If success, end
            "failure": END, # If failure, end
            "max_iterations_reached": END # If max iterations reached, end
        }
    )
    logger.info("Added conditional edges from check_mission_status")

    # The update_mission_state node transitions to the dynamically assigned role
    graph.add_edge("update_mission_state", dynamic_role_assignment)
    logger.info("Added edge: source=update_mission_state, target=dynamic_role_assignment")

    # Compile the graph
    start_time_compile = time.perf_counter()
    compiled_graph = graph.compile()
    end_time_compile = time.perf_counter()
    logger.info("LangGraph compilation complete: duration_ms=%s", (end_time_compile - start_time_compile)*1000)

    return compiled_graph