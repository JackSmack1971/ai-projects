import time
import os # Added os import for scenario file loading
import json # Added json import for scenario file loading
from typing import Any

# Import centralized logging configuration
from logging_config import logger

# Import Config
from config_loader import Config

# Import MissionState
from graph_builder import MissionState

# Import metrics
from main import MISSIONS_STARTED_TOTAL, MISSIONS_COMPLETED_TOTAL

def run_scenario(scenario: str, compiled_graph: Any) -> None:
    """Runs a single mission scenario using the compiled LangGraph."""
    initial_state = MissionState(messages=[scenario]) # Initialize state with the scenario brief
    bound_logger = logger.bind(request_id=initial_state.request_id) # Bind request_id to logger for this scenario

    bound_logger.info("Mission simulation started: request_id=%s, scenario=%s", initial_state.request_id, scenario)
    MISSIONS_STARTED_TOTAL.inc() # Metrics: Increment mission started counter

    start_time = time.perf_counter()
    try:
        # Run the compiled graph
        # The .invoke() method takes the initial state and runs the graph until it reaches an END node
        final_state = compiled_graph.invoke(initial_state, {"recursion_limit": Config.MAX_ITERATIONS})

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Check if the mission ended due to reaching the iteration limit
        if final_state.mission_status == "in_progress" and len(final_state.mission_progress) >= Config.MAX_ITERATIONS:
             final_state.mission_status = "max_iterations_reached"
             bound_logger.warning("Mission reached maximum iterations without concluding.", request_id=initial_state.request_id, max_iterations=Config.MAX_ITERATIONS)


        MISSIONS_COMPLETED_TOTAL.labels(status=final_state.mission_status).inc() # Metrics: Increment mission completed counter with status

        bound_logger.info("Mission simulation ended: final_status=%s, duration_ms=%s", final_state.mission_status, duration*1000)
        # Print final messages or summary
        print("\n--- Mission Summary ---")
        print(f"Scenario: {scenario}")
        print(f"Final Status: {final_state.mission_status}")
        print("Mission Progress:")
        for entry in final_state.mission_progress:
            print(f"- {entry}")
        print("--- End Summary ---\n")


    except Exception as e:
        bound_logger.error("Error occurred during scenario execution: error=%s", str(e))
        final_state = initial_state # Use initial state to capture request_id
        final_state.mission_status = "failure" # Mark mission as failure
        MISSIONS_COMPLETED_TOTAL.labels(status=final_state.mission_status).inc() # Metrics: Increment mission completed counter with status
        end_time = time.perf_counter() # Capture end time even on error
        duration = end_time - start_time
        bound_logger.info("Mission simulation ended: final_status=%s, duration_ms=%s", final_state.mission_status, duration*1000)
        print("\n--- Mission Summary ---")
        print(f"Scenario: {scenario}")
        print(f"Final Status: {final_state.mission_status}")
        print("Mission Progress:")
        for entry in final_state.mission_progress:
            print(f"- {entry}")
        print("--- End Summary ---\n")