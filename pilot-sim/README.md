# Project Documentation

This project is a system designed for managing and executing scenarios involving agents and roles, likely for simulations, testing, or automated workflows.

## SAPPO Concepts

*   **:ArchitecturalPattern**: The project follows a modular architecture. Key components include configuration loading, graph building, scenario execution, agent creation, and role management. There is no explicit :RecursiveAlgorithm identified in the top-level structure, but individual components might employ recursion internally.
*   **:Technology choices**: The project is primarily built using Python. Configuration and scenario data are handled using JSON files.
*   **:Context**: The operational context involves defining scenarios (`scenarios/`), configuring the system (`config.py`, `.env`), defining roles (`roles.json`), creating agents (`agent_factory.py`), managing roles (`role_manager.py`), building execution graphs (`graph_builder.py`), and running scenarios (`scenario_runner.py`).

## Project Structure

```
.
├── .env                 # Environment variables (e.g., API keys)
├── agent_factory.py     # Handles creation of agents
├── config.py            # Configuration loading and management
├── graph_builder.py     # Builds execution graphs for scenarios
├── logging_config.py    # Configuration for logging
├── main.py              # Main entry point of the application
├── README.md            # Project documentation
├── role_manager.py      # Manages roles and their assignments
├── roles.json           # Defines available roles
├── scenario_runner.py   # Executes defined scenarios
├── scenarios/           # Directory containing scenario definitions
│   └── ocean_guardian.json # Example scenario file
├── storage/             # Directory for data storage (e.g., vector stores)
│   └── vectorstore_manager.py # Manages vector store interactions
└── tests/               # Directory for project tests
    └── test_agent_initialization.py # Example test file
```

## Components

*   `main.py`: The main script to run the application.
*   `config.py`: Loads and provides access to project configuration.
*   `agent_factory.py`: Responsible for creating agent instances.
*   `role_manager.py`: Manages the definition and assignment of roles to agents.
*   `graph_builder.py`: Constructs the operational graph based on scenario definitions.
*   `scenario_runner.py`: Orchestrates the execution of a scenario based on the built graph.
*   `logging_config.py`: Sets up the logging for the application.
*   `storage/vectorstore_manager.py`: Handles interactions with vector databases for data storage or retrieval.

## Setup

1.  Clone the repository.
2.  Create a `.env` file based on required environment variables (details should be provided separately or in a template file).
3.  Ensure `roles.json` is configured with the necessary role definitions.
4.  Place scenario definition files in the `scenarios/` directory.

## Targeted Testing Strategy

The project likely employs a targeted testing strategy focusing on:

*   **CORE LOGIC TESTING**: Unit tests for individual Python modules (`config.py`, `agent_factory.py`, `role_manager.py`, etc.) to verify their core functions and logic in isolation. This ensures that each component behaves as expected before integration.
*   **CONTEXTUAL INTEGRATION TESTING**: Tests that verify the interactions between different components. Examples include:
    *   Loading a configuration and ensuring it's correctly used by other modules.
    *   Creating agents with specific roles and verifying their behavior within a simple scenario context.
    *   Building a graph from a scenario definition and confirming its structure is correct.

This targeted approach allows for quick feedback on the correctness of individual units and key integration points without requiring a full end-to-end test suite for every change.