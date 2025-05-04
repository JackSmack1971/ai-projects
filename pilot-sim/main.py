from prometheus_client import Counter, Histogram
from http.server import BaseHTTPRequestHandler, HTTPServer
from prometheus_client.exposition import generate_latest, REGISTRY
import os
import time
import json
import uuid
import logging
import threading
from typing import List, Dict, Any, Optional, Callable

import structlog
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

import pybreaker
from pydantic import BaseModel, ValidationError

# Import vectorstore management function
from storage.vectorstore_manager import process_pdf_and_create_vectorstore

# Import centralized logging configuration
from logging_config import configure_logging, logger

# Import refactored modules
from config import config
from role_manager import AgentRole, ROLE_REGISTRY, load_roles_from_config, role_reloader_thread, get_role
from agent_factory import create_llm, AGENTS, AGENT_NODES, create_agent, initialize_agents, create_agent_node, initialize_agent_nodes
from graph_builder import MissionState, AgentOutput, dynamic_role_assignment, check_mission_status_node, update_mission_state_node, create_command_control_graph
from scenario_runner import run_scenario

# --- Circuit Breaker Setup ---
rag_breaker = pybreaker.CircuitBreaker(
    fail_max=config.circuit_breaker_fail_threshold,
    reset_timeout=config.circuit_breaker_reset_timeout,
    exclude=config.circuit_breaker_exclude_exceptions
)

agent_breaker = pybreaker.CircuitBreaker(
    fail_max=config.circuit_breaker_fail_threshold,
    reset_timeout=config.circuit_breaker_reset_timeout,
    exclude=config.circuit_breaker_exclude_exceptions
)

# --- Metrics Setup ---
# Counters
MISSIONS_STARTED_TOTAL = Counter('missions_started_total', 'Total number of missions started')
MISSIONS_COMPLETED_TOTAL = Counter('missions_completed_total', 'Total number of missions completed', ['status'])
AGENT_STEPS_EXECUTED_TOTAL = Counter('agent_steps_executed_total', 'Total number of agent steps executed', ['agent_name'])
TOOL_USES_TOTAL = Counter('tool_uses_total', 'Total number of tool uses', ['tool_name'])

# Histograms
MISSION_DURATION_SECONDS = Histogram('mission_duration_seconds', 'Mission duration in seconds')
AGENT_STEP_DURATION_SECONDS = Histogram('agent_step_duration_seconds', 'Agent step duration in seconds', ['agent_name'])
TOOL_USE_DURATION_SECONDS = Histogram('tool_use_duration_seconds', 'Tool use duration in seconds', ['tool_name'])

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
            self.end_headers()
            self.wfile.write(generate_latest(REGISTRY))
        else:
            self.send_response(404)
            self.end_headers()

def start_metrics_server(port: int):
    """Starts a simple HTTP server to expose Prometheus metrics."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, MetricsHandler)
    logger.info("Starting metrics server: port=%s", port)
    # Run the server in a separate thread so it doesn't block the main application
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True # Allow the main program to exit even if the thread is running
    server_thread.start()
    logger.info("Metrics server started in background thread")

# --- Vectorstore Management ---
# Make handbook_rag_chain globally accessible after setup
handbook_rag_chain = None

@retry(stop=stop_after_attempt(config.max_retries), wait=wait_exponential(multiplier=1, min=4, max=10))
def setup_rag_chain(pdf_url: str, collection_name: str, vectorstore_path: str, embedding_model_name: str, api_key: str) -> Optional[Any]:
    """
    Set up a Retrieval-Augmented Generation (RAG) chain.

    Args:
        pdf_url (str): URL of the PDF to be processed.
        collection_name (str): Name of the collection for vector storage.
        vectorstore_path (str): Path to persist the vectorstore.
        embedding_model_name (str): Name of the embedding model to use.
        api_key (str): OpenRouter API key.

    Returns:
        Optional[Any]: The RAG chain if successful, None otherwise.

    Raises:
        RetryError: If the operation fails after maximum retries.
    """
    logger.info("Attempting to set up RAG chain: pdf_url=%s, collection_name=%s, vectorstore_path=%s", pdf_url, collection_name, vectorstore_path)
    vectorstore = process_pdf_and_create_vectorstore(pdf_url, collection_name, vectorstore_path, embedding_model_name)
    if not vectorstore:
        logger.error("Failed to create or load vector store.: pdf_url=%s, collection_name=%s, vectorstore_path=%s", pdf_url, collection_name, vectorstore_path)
        return None
    logger.info("Successfully created or loaded vector store.: collection_name=%s, vectorstore_path=%s", collection_name, vectorstore_path)

    retriever = vectorstore.as_retriever()
    logger.info("Successfully set up retriever from vector store: collection_name=%s", collection_name)

    # Advanced Prompt Engineering: Instruct for step-by-step reasoning and JSON output
    rag_prompt = ChatPromptTemplate.from_template(
        """Context: {context}

Query: {question}

Instructions:
1. Analyze the context provided.
2. Think step-by-step to arrive at the answer based *only* on the context.
3. Provide your step-by-step reasoning in the 'reasoning' field of the JSON output.
4. Provide the final answer based on your reasoning in the 'final_answer' field of the JSON output.
5. If the context indicates the mission is complete (e.g., objective achieved, error encountered), set the 'mission_status' field to "success" or "failure". Otherwise, leave it as "in_progress".
6. Your output MUST be a JSON object matching the following structure:
   ```json
   {{
     "reasoning": "...",
     "final_answer": "...",
     "mission_status": "in_progress" | "success" | "failure"
   }}
   ```
7. Ensure the JSON is valid and can be parsed directly.
8. Do not include any other text outside the JSON block.

Example:
Context: The Air Force Handbook states that pilots must maintain situational awareness at all times.
Query: What is a key responsibility of a pilot according to the handbook?
```json
{{
  "reasoning": "The context explicitly states that pilots must maintain situational awareness.",
  "final_answer": "According to the Air Force Handbook, a key responsibility of a pilot is to maintain situational awareness at all times.",
  "mission_status": "in_progress"
}}
```

Reasoning:""" # The LLM should start its response with the JSON object
    )

    rag_chain = (
        {"context": lambda x: retriever.invoke(x["question"]), "question": lambda x: x["question"]}
        | rag_prompt
        | create_llm(api_key)
        # We will parse the JSON output in the agent step, not here.
        # The agent prompt will guide the LLM to produce JSON.
        | StrOutputParser() # Still output as string for agent to parse
    )

    # TODO: Add unit tests for setup_rag_chain and the rag_chain itself.
    # Test cases should cover successful setup, failure to load/create vectorstore,
    # and expected output format from the chain given sample context and queries.

    logger.info("RAG chain setup complete.: collection_name=%s", collection_name)

    return rag_chain

# --- Tools ---
@tool("retrieve_information")
@retry(stop=stop_after_attempt(config.max_retries), wait=wait_exponential(multiplier=1, min=4, max=10))
def retrieve_information(query: str, request_id: str) -> str:
    """
    Retrieve information from the 'Air Force Handbook'.

    Args:
        query (str): The query to search for in the handbook.
        request_id (str): The unique identifier for the current request/trace.

    Returns:
        str: The retrieved information or an error message.
    """
    start_time = time.perf_counter()
    TOOL_USES_TOTAL.labels(tool_name="retrieve_information").inc() # Metrics: Increment tool use counter
    global handbook_rag_chain # Declare global to access the chain
    if not handbook_rag_chain:
        logger.error("RAG chain is not initialized.: request_id=%s", request_id)
        return config.fallback_message

    logger.info("Using retrieve_information tool: request_id=%s, query=%s", request_id, query)
    try:
        # Integrate circuit breaker logic around this external call.
        @rag_breaker
        def invoke_rag_chain(chain, input_data):
            logger.debug("Attempting to invoke RAG chain via circuit breaker.: request_id=%s", request_id)
            return chain.invoke(input_data)

        logger.debug("Before invoking RAG chain via circuit breaker.: request_id=%s", request_id)
        result = invoke_rag_chain(handbook_rag_chain, {"question": query})
        logger.debug("After invoking RAG chain via circuit breaker.: request_id=%s", request_id)
        end_time = time.perf_counter()
        duration = end_time - start_time
        TOOL_USE_DURATION_SECONDS.labels(tool_name="retrieve_information").observe(duration) # Metrics: Observe tool use duration
        logger.info("retrieve_information tool successful: request_id=%s, query=%s, duration_ms=%s", request_id, query, duration*1000)
        logger.info("RAG chain invocation successful.: request_id=%s", request_id)
        return result
    except pybreaker.CircuitBreakerError as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        TOOL_USE_DURATION_SECONDS.labels(tool_name="retrieve_information").observe(duration) # Metrics: Observe tool use duration even on circuit breaker trip
        logger.warning("Circuit breaker tripped for retrieve_information tool.: request_id=%s, error=%s, duration_ms=%s", request_id, str(e), duration*1000)
        return config.fallback_message
    except RetryError as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        TOOL_USE_DURATION_SECONDS.labels(tool_name="retrieve_information").observe(duration) # Metrics: Observe tool use duration even on failure
        logger.error("Retry attempts exhausted for retrieving information: request_id=%s, query=%s, error=%s, duration_ms=%s", request_id, query, str(e), duration*1000)
        return config.fallback_message
    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        TOOL_USE_DURATION_SECONDS.labels(tool_name="retrieve_information").observe(duration) # Metrics: Observe tool use duration even on error
        logger.error("Error retrieving information: request_id=%s, query=%s, error=%s, duration_ms=%s", request_id, query, str(e), duration*1000)
        return config.fallback_message

# Mapping of tool names (strings) to tool functions
TOOL_REGISTRY: Dict[str, Callable] = {
    "retrieve_information": retrieve_information
}

# --- Main Execution ---
def main() -> None:
    """Main function to initialize and run the application."""
    global handbook_rag_chain # Declare global to assign the chain

    # 1. Configure logging
    configure_logging()
    logger.info("Application started.")

    try:
        # 2. Configuration is loaded automatically on import of config.py
        #    No explicit initialization function needed here.
        logger.info("Application configuration loaded.")

        # 3. Start metrics server in a background thread
        start_metrics_server(config.metrics_port)

        # 4. Setup RAG chain (Vectorstore)
        handbook_rag_chain = setup_rag_chain(
            pdf_url=config.pdf_url,
            collection_name=config.collection_name,
            vectorstore_path=config.vectorstore_path,
            embedding_model_name=config.embedding_model,
            api_key=config.api_key # Pass API key explicitly
        )
        if not handbook_rag_chain:
            logger.error("Failed to set up RAG chain. Exiting.")
            return # Exit if RAG setup fails

        # 5. Load agent roles from configuration file
        load_roles_from_config(config.roles_config_path, force_reload=True) # Force reload on startup

        # 6. Start role reloader thread
        role_reloader_thread_instance = threading.Thread(
            target=role_reloader_thread,
            args=(config.roles_config_path, config.roles_reload_interval_sec),
            daemon=True # Allow the main program to exit even if the thread is running
        )
        role_reloader_thread_instance.start()
        logger.info("Role reloader thread started.")

        # 7. Initialize agents and agent nodes
        initialize_agents(config.api_key) # Pass API key explicitly
        initialize_agent_nodes()

        # 8. Create the command and control graph
        compiled_graph = create_command_control_graph()
        logger.info("Command and control graph created.")

        # 9. Load and run scenarios
        scenarios_dir = config.scenarios_dir
        if not os.path.exists(scenarios_dir):
            logger.warning("Scenarios directory not found: path=%s", scenarios_dir)
            return # Exit if scenarios directory doesn't exist

        logger.info("Loading scenarios from directory: path=%s", scenarios_dir)
        scenario_files = [f for f in os.listdir(scenarios_dir) if f.endswith('.json')]

        if not scenario_files:
            logger.warning("No scenario files found in directory: path=%s", scenarios_dir)
            return # Exit if no scenario files are found

        for filename in scenario_files:
             # Validate and sanitize scenario_path
             scenario_path = os.path.join(config.scenarios_dir, filename)
             base_dir = os.path.abspath(config.scenarios_dir)
             abs_scenario_path = os.path.abspath(scenario_path)
             normalized_scenario_path = os.path.normpath(abs_scenario_path)

             # Check if the normalized path is within the scenarios directory
             if not normalized_scenario_path.startswith(base_dir):
                 logger.error("Attempted to load scenario file from outside allowed directory: path=%s", scenario_path)
                 continue # Skip this file

             # Assuming each JSON file contains a single scenario string
             try:
                 with open(scenario_path, 'r') as f:
                     scenario_data = json.load(f)
                     # Assuming the scenario is stored under a key like "scenario"
                     scenario_text = scenario_data.get("scenario")
                     if scenario_text:
                         logger.info("Running scenario from file: filename=%s", filename)
                         run_scenario(scenario_text, compiled_graph)
                     else:
                         logger.warning("Scenario data not found in file: filename=%s", filename)
             except json.JSONDecodeError:
                 logger.error("Error decoding scenario JSON file: filename=%s", filename)
             except Exception as e:
                 logger.error("An unexpected error occurred while processing scenario file: filename=%s, error=%s", filename, str(e))


    except ValueError as e:
        logger.error("Configuration error: %s", str(e))
    except Exception as e:
        logger.error("An unexpected error occurred during application startup: %s", str(e))

    logger.info("Application finished.")

if __name__ == "__main__":
    main()