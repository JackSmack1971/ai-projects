import json
import time
from typing import Dict, Callable, Any, List, Optional
from functools import lru_cache

import pybreaker
from pydantic import BaseModel, ValidationError
from prometheus_client import Counter, Histogram # Import metrics types

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool # Keep tool import for now, though tools might be defined elsewhere later
from langchain_core.output_parsers import StrOutputParser # Not directly used here, but might be needed for agent output processing
from langchain_core.runnables import RunnablePassthrough # Not directly used here

# Import centralized logging configuration
from logging_config import logger

# Import Config and AgentRole (will be needed)
from config import config
from role_manager import AgentRole, ROLE_REGISTRY

# Import MissionState and AgentOutput
from graph_builder import MissionState, AgentOutput

# --- LLM Setup ---
@lru_cache(maxsize=1)
def create_llm(api_key: str, model: str = config.LLM_MODEL) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for OpenRouter."""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

# --- Agent Creation and Execution ---
AGENTS: Dict[str, AgentExecutor] = {}
AGENT_NODES: Dict[str, Callable] = {}

def create_agent(llm: ChatOpenAI, role: AgentRole) -> AgentExecutor:
    """
    Create an agent with the given LLM and role configuration.

    Args:
        llm (ChatOpenAI): The language model to use.
        role (AgentRole): The role configuration for the agent.

    Returns:
        AgentExecutor: The created agent executor.
    """
    # Instruct the agent to output in the specified JSON format
    system_prompt_with_json_instruction = role.system_prompt + """

Your response MUST be a JSON object matching the following structure:
```json
{
  "reasoning": "...",
  "final_answer": "...",
  "mission_status": "in_progress" | "success" | "failure"
}
```
Ensure the JSON is valid and can be parsed directly. Do not include any other text outside the JSON block.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_with_json_instruction),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_react_agent(llm, role.tools, prompt)
    # handle_parsing_errors is useful, but we'll add our own explicit validation
    return AgentExecutor(agent=agent, tools=role.tools, handle_parsing_errors=False)

def initialize_agents(api_key: str) -> None:
    """Initialize agents based on the role registry."""
    llm = create_llm(api_key)
    AGENTS.clear() # Clear existing agents for hot-reloading
    for role_name, role_config in ROLE_REGISTRY.items():
        AGENTS[role_name] = create_agent(llm, role_config)
        logger.info("Initialized agent: agent_name=%s", role_name)

def create_agent_node(
    agent_name: str,
    agent_steps_executed_total: Counter,
    agent_step_duration_seconds: Histogram,
    agent_breaker: pybreaker.CircuitBreaker
) -> Callable:
    """Create a LangGraph node function for a given agent."""
    agent_executor = AGENTS.get(agent_name)
    if not agent_executor:
        raise ValueError(f"Agent '{agent_name}' not found in registry.")

    def agent_node(state: MissionState) -> MissionState:
        """
        Execute the agent and update the state.
        """
        start_time = time.perf_counter()
        agent_steps_executed_total.labels(agent_name=agent_name).inc() # Metrics: Increment agent step counter
        logger.info("Executing agent node: agent_name=%s, request_id=%s", agent_name, state.request_id)

        # Input Validation: Ensure state.messages is a list
        if not isinstance(state.messages, list):
            logger.error("Invalid input: state.messages is not a list", agent_name=agent_name, request_id=state.request_id, messages_type=type(state.messages))
            state.mission_progress.append(f"{agent_name}: Invalid input received.")
            state.mission_status = "failure"
            return state

        try:
            # Integrate circuit breaker logic around agent execution.
            @agent_breaker
            def invoke_agent(executor, inputs):
                 # Langchain agents expect a dictionary with 'input' and 'intermediate_steps'
                 # We need to format the messages from the state into a string for the agent's input
                 # Or, if using a chat agent, pass the message history directly.
                 # Assuming a simple string input for now based on the prompt structure.
                 # A more complex agent might need a different input format.
                 # TODO: Address the issue of only processing the last message (later task)
                 agent_input = {"input": state.messages[-1] if state.messages else "", "intermediate_steps": []} # Simple input from last message
                 return executor.invoke(agent_input)

            # The agent's output is expected to be a JSON string based on the prompt instructions
            agent_output_str = invoke_agent(agent_executor, state) # Pass the state directly for now, adjust if agent input needs specific formatting

            # Attempt to parse the JSON output
            try:
                # Output Validation: Attempt to parse and validate with Pydantic model
                agent_output_data = json.loads(agent_output_str)
                parsed_output = AgentOutput(**agent_output_data) # Validate with Pydantic model

                # Update state based on parsed output
                state.mission_progress.append(f"{agent_name}: {parsed_output.final_answer}")
                if parsed_output.mission_status and parsed_output.mission_status != "in_progress":
                     state.mission_status = parsed_output.mission_status
                     logger.info("Agent suggested mission status change", agent_name=agent_name, new_status=state.mission_status, request_id=state.request_id)

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error("Agent output parsing or validation failed", agent_name=agent_name, output=agent_output_str, error=str(e), request_id=state.request_id)
                state.mission_progress.append(f"{agent_name}: Error processing output.")
                state.mission_status = "failure" # Mark mission as failure on parsing error


            end_time = time.perf_counter()
            duration = end_time - start_time
            agent_step_duration_seconds.labels(agent_name=agent_name).observe(duration) # Metrics: Observe agent step duration
            logger.info("Agent node execution complete", agent_name=agent_name, request_id=state.request_id, duration_ms=duration*1000)
            return state # Return the updated state

        except pybreaker.CircuitBreakerError as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            agent_step_duration_seconds.labels(agent_name=agent_name).observe(duration) # Metrics: Observe agent step duration even on circuit breaker trip
            logger.warning("Circuit breaker tripped for agent execution.", agent_name=agent_name, error=str(e), duration_ms=duration*1000, request_id=state.request_id)
            state.mission_progress.append(f"{agent_name}: Operation failed due to system overload. Please try again.")
            state.mission_status = "failure" # Mark mission as failure on circuit breaker trip
            return state # Return the updated state
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            agent_step_duration_seconds.labels(agent_name=agent_name).observe(duration) # Metrics: Observe agent step duration even on error
            logger.error("Error executing agent node", agent_name=agent_name, error=str(e), duration_ms=duration*1000, request_id=state.request_id)
            state.mission_progress.append(f"{agent_name}: An unexpected error occurred.")
            state.mission_status = "failure" # Mark mission as failure on unexpected error
            return state # Return the updated state


    return agent_node


def initialize_agent_nodes(
    agent_steps_executed_total: Counter,
    agent_step_duration_seconds: Histogram,
    agent_breaker: pybreaker.CircuitBreaker
) -> None:
    """Initialize agent nodes from the role registry."""
    AGENT_NODES.clear() # Clear existing nodes for hot-reloading
    for role_name in ROLE_REGISTRY.keys():
        AGENT_NODES[role_name] = create_agent_node(
            role_name,
            agent_steps_executed_total,
            agent_step_duration_seconds,
            agent_breaker
        )
        logger.info("Initialized agent node: node_name=%s", role_name)