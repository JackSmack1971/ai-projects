import unittest
from unittest.mock import patch, mock_open
import os
from config import AppConfig

class TestAppConfig(unittest.TestCase):

    @patch('os.getenv')
    @patch('dotenv.load_dotenv')
    def test_appconfig_init_success(self, mock_load_dotenv, mock_getenv):
        """Test AppConfig initialization with all required env vars set."""
        mock_getenv.side_effect = lambda x, default=None: {
            "API_KEY": "fake_api_key",
            "LOG_LEVEL": "DEBUG",
            "FALLBACK_MESSAGE": "Custom fallback",
            "CIRCUIT_BREAKER_FAIL_THRESHOLD": "10",
            "CIRCUIT_BREAKER_RESET_TIMEOUT": "60",
            "MAX_RETRIES": "5",
            "METRICS_PORT": "9000",
            "PDF_URL": "http://fakeurl.com/fake.pdf",
            "COLLECTION_NAME": "fake_collection",
            # Add other required env vars from __init__ if any
        }.get(x, default)

        config = AppConfig()

        self.assertEqual(config.api_key, "fake_api_key")
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.fallback_message, "Custom fallback")
        self.assertEqual(config.circuit_breaker_fail_threshold, 10)
        self.assertEqual(config.circuit_breaker_reset_timeout, 60)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.metrics_port, 9000)
        self.assertEqual(config.pdf_url, "http://fakeurl.com/fake.pdf")
        self.assertEqual(config.collection_name, "fake_collection")
        mock_load_dotenv.assert_called_once()

    @patch('os.getenv')
    @patch('dotenv.load_dotenv')
    def test_appconfig_init_defaults(self, mock_load_dotenv, mock_getenv):
        """Test AppConfig initialization with default values."""
        mock_getenv.side_effect = lambda x, default=None: {
            "API_KEY": "fake_api_key", # Required, must be set
            "PDF_URL": "http://fakeurl.com/fake.pdf", # Required, must be set
            "COLLECTION_NAME": "fake_collection", # Required, must be set
            # Omit optional env vars to check defaults
        }.get(x, default)

        config = AppConfig()

        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.fallback_message, "An error occurred or information could not be retrieved.")
        self.assertEqual(config.circuit_breaker_fail_threshold, 5)
        self.assertEqual(config.circuit_breaker_reset_timeout, 30)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.metrics_port, 8000)
        mock_load_dotenv.assert_called_once()

    @patch('os.getenv')
    @patch('dotenv.load_dotenv')
    def test_appconfig_init_missing_required(self, mock_load_dotenv, mock_getenv):
        """Test AppConfig initialization when a required env var is missing."""
        mock_getenv.side_effect = lambda x, default=None: {
            # Omit API_KEY
            "LOG_LEVEL": "DEBUG",
            "PDF_URL": "http://fakeurl.com/fake.pdf",
            "COLLECTION_NAME": "fake_collection",
        }.get(x, default)

        with self.assertRaises(ValueError) as cm:
            AppConfig()

        self.assertIn("A required configuration value is not set.", str(cm.exception))
        mock_load_dotenv.assert_called_once()

    @patch('os.getenv')
    def test_get_required_env_success(self, mock_getenv):
        """Test _get_required_env when variable is set."""
        mock_getenv.return_value = "some_value"
        config = AppConfig() # Need an instance to call the method
        # Patch the method directly for isolated testing
        with patch.object(config, '_get_required_env', return_value="some_value") as mock_method:
             result = config._get_required_env("TEST_VAR")
             self.assertEqual(result, "some_value")
             mock_method.assert_called_once_with("TEST_VAR")


    @patch('os.getenv')
    def test_get_required_env_missing(self, mock_getenv):
        """Test _get_required_env when variable is missing."""
        mock_getenv.return_value = None
        config = AppConfig() # Need an instance
        # Patch the method directly for isolated testing
        with patch.object(config, '_get_required_env', side_effect=ValueError("A required configuration value is not set.")) as mock_method:
            with self.assertRaises(ValueError) as cm:
                config._get_required_env("MISSING_VAR")
            self.assertIn("A required configuration value is not set.", str(cm.exception))
            mock_method.assert_called_once_with("MISSING_VAR")

    @patch('os.path.abspath')
    @patch('os.path.normpath')
    def test_validate_path_within_workspace_valid(self, mock_normpath, mock_abspath):
        """Test _validate_path_within_workspace with a valid path."""
        mock_abspath.side_effect = lambda x: f"/fake/workspace/{x}" if x != "." else "/fake/workspace"
        mock_normpath.side_effect = lambda x: x # Assume normalization doesn't change for this test

        config = AppConfig() # Need an instance
        # Patch the method directly for isolated testing
        with patch.object(config, '_validate_path_within_workspace') as mock_method:
            try:
                config._validate_path_within_workspace("some/path", "SOME_VAR")
                mock_method.assert_called_once_with("some/path", "SOME_VAR")
            except ValueError:
                self.fail("_validate_path_within_workspace raised ValueError unexpectedly")


    @patch('os.path.abspath')
    @patch('os.path.normpath')
    def test_validate_path_within_workspace_invalid(self, mock_normpath, mock_abspath):
        """Test _validate_path_within_workspace with an invalid path."""
        mock_abspath.side_effect = lambda x: f"/outside/workspace/{x}" if x != "." else "/fake/workspace"
        mock_normpath.side_effect = lambda x: x # Assume normalization doesn't change for this test

        config = AppConfig() # Need an instance
        # Patch the method directly for isolated testing
        with patch.object(config, '_validate_path_within_workspace', side_effect=ValueError("Configured path 'SOME_VAR' is outside the allowed workspace directory: outside/path")) as mock_method:
            with self.assertRaises(ValueError) as cm:
                config._validate_path_within_workspace("outside/path", "SOME_VAR")
            self.assertIn("Configured path 'SOME_VAR' is outside the allowed workspace directory: outside/path", str(cm.exception))
            mock_method.assert_called_once_with("outside/path", "SOME_VAR")

    def test_validate_url_valid(self):
        """Test _validate_url with a valid URL."""
        config = AppConfig() # Need an instance
        # Patch the method directly for isolated testing
        with patch.object(config, '_validate_url') as mock_method:
            try:
                config._validate_url("http://valid.com", "VALID_URL")
                config._validate_url("https://another.org/path", "ANOTHER_URL")
                mock_method.assert_any_call("http://valid.com", "VALID_URL")
                mock_method.assert_any_call("https://another.org/path", "ANOTHER_URL")
            except ValueError:
                self.fail("_validate_url raised ValueError unexpectedly")

    def test_validate_url_invalid_format(self):
        """Test _validate_url with an invalid URL format."""
        config = AppConfig() # Need an instance
        # Patch the method directly for isolated testing
        with patch.object(config, '_validate_url', side_effect=ValueError("Configured URL 'INVALID_URL' validation failed: Configured URL 'INVALID_URL' is not a valid URL: invalid-url")) as mock_method:
            with self.assertRaises(ValueError) as cm:
                config._validate_url("invalid-url", "INVALID_URL")
            self.assertIn("Configured URL 'INVALID_URL' validation failed: Configured URL 'INVALID_URL' is not a valid URL: invalid-url", str(cm.exception))
            mock_method.assert_called_once_with("invalid-url", "INVALID_URL")

    def test_validate_url_disallowed_scheme(self):
        """Test _validate_url with a disallowed URL scheme."""
        config = AppConfig() # Need an instance
        # Patch the method directly for isolated testing
        with patch.object(config, '_validate_url', side_effect=ValueError("Configured URL 'FTP_URL' validation failed: Configured URL 'FTP_URL' uses an disallowed scheme: ftp")) as mock_method:
            with self.assertRaises(ValueError) as cm:
                config._validate_url("ftp://ftp.com", "FTP_URL")
            self.assertIn("Configured URL 'FTP_URL' validation failed: Configured URL 'FTP_URL' uses an disallowed scheme: ftp", str(cm.exception))
            mock_method.assert_called_once_with("ftp://ftp.com", "FTP_URL")

if __name__ == '__main__':
    unittest.main()