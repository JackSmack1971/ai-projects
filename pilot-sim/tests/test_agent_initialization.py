import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json

# Assuming main.py is in the parent directory relative to the tests directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import load_roles_from_config, create_agent, initialize_agents, initialize_agent_nodes, AgentRole, Config, ROLE_REGISTRY, AGENTS, AGENT_NODES, TOOL_REGISTRY
import main # Import main module to access module-level variables

# Mock the logger to prevent excessive output during tests
patch('main.logger', MagicMock()).start()
patch('storage.vectorstore_manager.logger', MagicMock()).start()
patch('logging_config.logger', MagicMock()).start()


class TestAgentInitialization(unittest.TestCase):

    def setUp(self):
        # Reset registries and agents before each test
        ROLE_REGISTRY.clear()
        AGENTS.clear()
        AGENT_NODES.clear()
        # Ensure a default retrieve_information tool exists for testing
        if "retrieve_information" not in TOOL_REGISTRY:
             TOOL_REGISTRY["retrieve_information"] = MagicMock()


    def tearDown(self):
        # Clean up after each test if necessary
        pass

    # --- Core Logic Tests ---

    @patch('main.open', new_callable=mock_open, read_data='''
[
    {
        "name": "Pilot",
        "system_prompt": "You are a pilot.",
        "tools": ["retrieve_information"],
        "dependencies": []
    },
    {
        "name": "Copilot",
        "system_prompt": "You are a copilot.",
        "tools": [],
        "dependencies": ["pilot"]
    }
]
''')
    @patch('main.os.path.getmtime', return_value=12345.0)
    @patch('main._roles_config_last_modified', None)
    def test_load_roles_from_config_success(self, mock_getmtime, mock_file):
        """Test successful loading of roles from a valid config file."""
        config_path = "roles.json"
        load_roles_from_config(config_path)

        self.assertEqual(len(ROLE_REGISTRY), 2)
        self.assertIn("pilot", ROLE_REGISTRY)
        self.assertIn("copilot", ROLE_REGISTRY)

        pilot_role = ROLE_REGISTRY["pilot"]
        self.assertEqual(pilot_role.name, "Pilot")
        self.assertEqual(pilot_role.system_prompt, "You are a pilot.")
        self.assertEqual(len(pilot_role.tools), 1)
        self.assertEqual(pilot_role.tools[0], TOOL_REGISTRY["retrieve_information"])
        self.assertEqual(pilot_role.dependencies, [])

        copilot_role = ROLE_REGISTRY["copilot"]
        self.assertEqual(copilot_role.name, "Copilot")
        self.assertEqual(copilot_role.system_prompt, "You are a copilot.")
        self.assertEqual(len(copilot_role.tools), 0)
        self.assertEqual(copilot_role.dependencies, ["pilot"])

    @patch('main.os.path.getmtime', side_effect=FileNotFoundError)
    @patch('main._roles_config_last_modified', None)
    def test_load_roles_from_config_file_not_found(self, mock_getmtime):
        """Test handling of FileNotFoundError when loading roles config."""
        config_path = "non_existent_roles.json"
        load_roles_from_config(config_path)

        # Should load default roles
        self.assertEqual(len(ROLE_REGISTRY), 3)
        self.assertIn("pilot", ROLE_REGISTRY)
        self.assertIn("copilot", ROLE_REGISTRY)
        self.assertIn("cso", ROLE_REGISTRY)
        self.assertIsNone(main._roles_config_last_modified) # Should reset modified time

    @patch('main.open', new_callable=mock_open, read_data='''
[
    {
        "name": "Pilot",
        "system_prompt": "You are a pilot.",
        "tools": ["retrieve_information"]
        // Missing dependencies field
    }
]
''')
    @patch('main.os.path.getmtime', return_value=12345.0)
    @patch('main._roles_config_last_modified', None)
    def test_load_roles_from_config_missing_dependencies(self, mock_getmtime, mock_file):
        """Test loading roles with missing optional fields like dependencies."""
        config_path = "roles.json"
        load_roles_from_config(config_path)

        self.assertEqual(len(ROLE_REGISTRY), 1)
        self.assertIn("pilot", ROLE_REGISTRY)

        pilot_role = ROLE_REGISTRY["pilot"]
        self.assertEqual(pilot_role.dependencies, []) # Should default to empty list

    @patch('main.open', new_callable=mock_open, read_data='''
[
    {
        "name": "Pilot",
        "system_prompt": "You are a pilot.",
        "tools": ["non_existent_tool"]
    }
]
''')
    @patch('main.os.path.getmtime', return_value=12345.0)
    @patch('main._roles_config_last_modified', None)
    def test_load_roles_from_config_unknown_tool(self, mock_getmtime, mock_file):
        """Test loading roles with an unknown tool."""
        config_path = "roles.json"
        load_roles_from_config(config_path)

        self.assertEqual(len(ROLE_REGISTRY), 1)
        self.assertIn("pilot", ROLE_REGISTRY)

        pilot_role = ROLE_REGISTRY["pilot"]
        self.assertEqual(len(pilot_role.tools), 0) # Unknown tool should not be added

    @patch('main.open', new_callable=mock_open, read_data='''
[
    {
        "name": "Pilot",
        // Missing system_prompt
        "tools": []
    }
]
''')
    @patch('main.os.path.getmtime', return_value=12345.0)
    @patch('main._roles_config_last_modified', None)
    def test_load_roles_from_config_missing_required_field(self, mock_getmtime, mock_file):
        """Test loading roles with a missing required field (system_prompt)."""
        config_path = "roles.json"
        load_roles_from_config(config_path)

        self.assertEqual(len(ROLE_REGISTRY), 0) # Role should be skipped

    @patch('main.open', new_callable=mock_open, read_data='''
[
    {
        "name": "Pilot",
        "system_prompt": "You are a pilot.",
        "tools": ["retrieve_information"]
    }
]
''')
    @patch('main.os.path.getmtime', return_value=12345.0)
    @patch('main._roles_config_last_modified', 12345.0) # Same modified time
    def test_load_roles_from_config_no_change(self, mock_getmtime, mock_last_modified):
        """Test that roles are not reloaded if the file hasn't changed."""
        config_path = "roles.json"
        # Populate registry before calling load
        ROLE_REGISTRY["existing_role"] = AgentRole(name="Existing", system_prompt="...", tools=[])
        load_roles_from_config(config_path)

        self.assertEqual(len(ROLE_REGISTRY), 1) # Should still have the existing role
        self.assertIn("existing_role", ROLE_REGISTRY)
        self.assertNotIn("pilot", ROLE_REGISTRY) # New role should not be loaded

    @patch('main.open', new_callable=mock_open, read_data='''
[
    {
        "name": "Pilot",
        "system_prompt": "You are a pilot.",
        "tools": ["retrieve_information"]
    }
]
''')
    @patch('main.os.path.getmtime', return_value=12346.0) # Different modified time
    @patch('main._roles_config_last_modified', 12345.0)
    @patch('main.initialize_application_config', return_value="mock_api_key")
    @patch('main.initialize_agents')
    @patch('main.initialize_agent_nodes')
    def test_load_roles_from_config_hot_reload(self, mock_init_nodes, mock_init_agents, mock_init_config, mock_getmtime, mock_last_modified, mock_file):
        """Test hot-reloading of roles when the file changes."""
        config_path = "roles.json"
        # Populate registry before calling load
        ROLE_REGISTRY["existing_role"] = AgentRole(name="Existing", system_prompt="...", tools=[])
        load_roles_from_config(config_path)

        self.assertEqual(len(ROLE_REGISTRY), 1) # Should clear and load new role
        self.assertNotIn("existing_role", ROLE_REGISTRY)
        self.assertIn("pilot", ROLE_REGISTRY)
        self.assertEqual(main._roles_config_last_modified, 12346.0) # Should update modified time
        mock_init_config.assert_called_once()
        mock_init_agents.assert_called_once_with("mock_api_key")
        mock_init_nodes.assert_called_once()

    @patch('main.os.path.getmtime', return_value=12345.0)
    @patch('main._roles_config_last_modified', None)
    def test_load_roles_from_config_json_decode_error(self, mock_getmtime, mock_file):
        """Test handling of JSONDecodeError when loading roles config."""
        config_path = "invalid.json"
        load_roles_from_config(config_path)

        # Should load default roles
        self.assertEqual(len(ROLE_REGISTRY), 3)
        self.assertIn("pilot", ROLE_REGISTRY)
        self.assertIn("copilot", ROLE_REGISTRY)
        self.assertIn("cso", ROLE_REGISTRY)
        self.assertIsNone(main._roles_config_last_modified) # Should reset modified time

    @patch('main.ChatOpenAI')
    def test_create_agent(self, MockChatOpenAI):
        """Test creation of an AgentExecutor."""
        mock_llm = MockChatOpenAI()
        mock_tool = MagicMock()
        role = AgentRole(name="TestAgent", system_prompt="Test prompt.", tools=[mock_tool], dependencies=[])

        # Mock create_react_agent to return a predictable object
        with patch('main.create_react_agent', return_value=MagicMock()) as mock_create_react_agent:
             # Mock AgentExecutor to return a predictable object
             with patch('main.AgentExecutor', return_value=MagicMock()) as mock_agent_executor:
                agent = create_agent(mock_llm, role)

                mock_create_react_agent.assert_called_once()
                # Check if the prompt includes the JSON instruction
                args, kwargs = mock_create_react_agent.call_args
                prompt_template = args[0]
                self.assertIn("Your response MUST be a JSON object matching the following structure:", prompt_template.messages[0].prompt.template)

                mock_agent_executor.assert_called_once_with(
                    agent=mock_create_react_agent.return_value,
                    tools=[mock_tool],
                    verbose=True, # Assuming verbose is True in the actual implementation or default
                    handle_parsing_errors=True, # Assuming handle_parsing_errors is True
                    max_iterations=Config.MAX_ITERATIONS # Check max_iterations
                )
                self.assertIsNotNone(agent)
                self.assertEqual(agent, mock_agent_executor.return_value)


    @patch('main.create_agent', return_value=MagicMock())
    @patch('main.initialize_application_config', return_value="mock_api_key")
    def test_initialize_agents(self, mock_init_config, mock_create_agent):
        """Test initialization of multiple agents."""
        # Populate ROLE_REGISTRY
        ROLE_REGISTRY["pilot"] = AgentRole(name="Pilot", system_prompt="...", tools=[], dependencies=[])
        ROLE_REGISTRY["copilot"] = AgentRole(name="Copilot", system_prompt="...", tools=[], dependencies=[])

        initialize_agents("mock_api_key")

        self.assertEqual(len(AGENTS), 2)
        self.assertIn("pilot", AGENTS)
        self.assertIn("copilot", AGENTS)
        self.assertEqual(mock_create_agent.call_count, 2)
        # Check if create_agent was called with the correct roles
        called_roles = [call.args[1].name.lower() for call in mock_create_agent.call_args_list]
        self.assertIn("pilot", called_roles)
        self.assertIn("copilot", called_roles)

    @patch('main.create_agent_node', side_effect=lambda name: f"node_for_{name}")
    def test_initialize_agent_nodes(self, mock_create_agent_node):
        """Test initialization of agent nodes."""
        # Populate AGENTS
        AGENTS["pilot"] = MagicMock()
        AGENTS["copilot"] = MagicMock()

        initialize_agent_nodes()

        self.assertEqual(len(AGENT_NODES), 2)
        self.assertIn("pilot", AGENT_NODES)
        self.assertIn("copilot", AGENT_NODES)
        self.assertEqual(AGENT_NODES["pilot"], "node_for_pilot")
        self.assertEqual(AGENT_NODES["copilot"], "node_for_copilot")
        mock_create_agent_node.assert_any_call("pilot")
        mock_create_agent_node.assert_any_call("copilot")
        self.assertEqual(mock_create_agent_node.call_count, 2)

    # --- Contextual Integration Tests ---

    @patch('main.initialize_application_config', return_value="mock_api_key")
    @patch('main.handbook_rag_chain', MagicMock()) # Mock the RAG chain
    def test_initialize_agents_with_rag_tool_integration(self, mock_init_config, mock_create_agent, mock_rag_chain):
        """
        Test that agents requiring the retrieve_information tool are initialized
        with the correct tool, which depends on the RAG chain.
        """
        # Ensure retrieve_information tool is in the registry and uses the mocked RAG chain
        mock_retrieve_info_tool = MagicMock()
        # Simulate the retrieve_information tool using the global handbook_rag_chain
        def mock_retrieve_information_func(query, request_id):
             if main.handbook_rag_chain:
                 return main.handbook_rag_chain.invoke({"question": query})
             return "RAG chain not initialized"

        TOOL_REGISTRY["retrieve_information"] = mock_retrieve_information_func


        # Populate ROLE_REGISTRY with a role that uses the RAG tool
        ROLE_REGISTRY["pilot"] = AgentRole(name="Pilot", system_prompt="...", tools=[TOOL_REGISTRY["retrieve_information"]], dependencies=[])
        ROLE_REGISTRY["copilot"] = AgentRole(name="Copilot", system_prompt="...", tools=[], dependencies=[])


        initialize_agents("mock_api_key")

        self.assertEqual(len(AGENTS), 2)
        self.assertIn("pilot", AGENTS)
        self.assertIn("copilot", AGENTS)

        # Verify that create_agent was called with the correct tools for each role
        pilot_call_args = None
        copilot_call_args = None
        for call_args, call_kwargs in mock_create_agent.call_args_list:
            role_arg = call_args[1]
            if role_arg.name.lower() == "pilot":
                pilot_call_args = call_args
            elif role_arg.name.lower() == "copilot":
                copilot_call_args = call_args

        self.assertIsNotNone(pilot_call_args)
        self.assertIsNotNone(copilot_call_args)

        # Check tools passed to create_agent for the pilot
        pilot_tools_passed = pilot_call_args[1].tools
        self.assertEqual(len(pilot_tools_passed), 1)
        self.assertEqual(pilot_tools_passed[0], TOOL_REGISTRY["retrieve_information"])

        # Check tools passed to create_agent for the copilot
        copilot_tools_passed = copilot_call_args[1].tools
        self.assertEqual(len(copilot_tools_passed), 0)

        # Further check: Although create_agent is mocked, in a real scenario,
        # the retrieve_information tool function would be passed. We can't
        # easily test if the *internal* logic of the tool (using handbook_rag_chain)
        # is correct here without more complex mocking or integration setup.
        # The focus of this test is verifying that the correct tool *object*
        # is passed to the agent creation logic based on the role configuration.
        # The test for the retrieve_information tool itself (including its RAG chain
        # interaction) would be a separate unit/integration test.


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)