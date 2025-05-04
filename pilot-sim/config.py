from urllib.parse import urlparse
import os
from dotenv import load_dotenv
from typing import Tuple, Type

# Load environment variables from .env file
load_dotenv()

class AppConfig:
    """
    Centralized configuration for the application.
    Loads values from environment variables and provides validation.
    """
    def __init__(self):
        # General
        self.api_key = self._get_required_env("API_KEY")
        self.log_level = os.getenv("LOG_LEVEL", "INFO") # Default to INFO
        self.fallback_message = os.getenv("FALLBACK_MESSAGE", "An error occurred or information could not be retrieved.") # Default fallback message

        # Circuit Breaker
        self.circuit_breaker_fail_threshold = int(os.getenv("CIRCUIT_BREAKER_FAIL_THRESHOLD", 5)) # Default to 5
        self.circuit_breaker_reset_timeout = int(os.getenv("CIRCUIT_BREAKER_RESET_TIMEOUT", 30)) # Default to 30 seconds
        # Note: CIRCUIT_BREAKER_EXCLUDE_EXCEPTIONS is harder to configure via env var.
        # For now, we'll keep a default or require code change if needed.
        # If needed, could parse a comma-separated string of exception names.
        self.circuit_breaker_exclude_exceptions: Tuple[Type[Exception], ...] = (
            # Add exceptions to exclude here, e.g., ValueError, TypeError
        )

        # Retries
        self.max_retries = int(os.getenv("MAX_RETRIES", 3)) # Default to 3

        # Metrics
        self.metrics_port = int(os.getenv("METRICS_PORT", 8000)) # Default to 8000

        # RAG/Vectorstore
        self.pdf_url = self._get_required_env("PDF_URL")
        self.collection_name = self._get_required_env("COLLECTION_NAME")
    def _get_required_env(self, var_name: str) -> str:
        """Gets an environment variable, raising an error if it's not set."""
        value = os.getenv(var_name)
        if not value:
            # Raise a generic ValueError to avoid leaking environment variable names
            raise ValueError("A required configuration value is not set.")
        return value

    def _validate_path_within_workspace(self, path: str, var_name: str):
        """Validates that a given path is within the current workspace directory."""
        base_dir = os.path.abspath(".")
        abs_path = os.path.abspath(path)
        normalized_path = os.path.normpath(abs_path)

        if not normalized_path.startswith(base_dir):
            raise ValueError(f"Configured path '{var_name}' is outside the allowed workspace directory: {path}")

    def _validate_url(self, url: str, var_name: str):
        """Validates that a given string is a well-formed URL with an allowed scheme."""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValueError(f"Configured URL '{var_name}' is not a valid URL: {url}")
            # Optional: Restrict schemes
            if result.scheme not in ['http', 'https']:
                 raise ValueError(f"Configured URL '{var_name}' uses an disallowed scheme: {result.scheme}")
        except ValueError as e:
            # Re-raise with a more informative message
            raise ValueError(f"Configured URL '{var_name}' validation failed: {e}")


# Create a single instance of the configuration to be imported elsewhere
config = AppConfig()

# Example usage (for testing purposes, can be removed later)
if __name__ == "__main__":
    try:
        print("Loading configuration...")
        print(f"API Key: {config.api_key[:4]}...") # Print partial key for security
        print(f"Log Level: {config.log_level}")
        print(f"Metrics Port: {config.metrics_port}")
        print(f"PDF URL: {config.pdf_url}")
        print(f"Scenarios Dir: {config.scenarios_dir}")
        print(f"Vectorstore Path: {config.vectorstore_path}")
        print(f"Roles Config Path: {config.roles_config_path}")
        print("Configuration loaded successfully.")
    except (EnvironmentError, ValueError) as e:
        print(f"Configuration Error: {e}")