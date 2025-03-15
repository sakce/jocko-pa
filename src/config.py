import json
import os
from typing import Any, Dict

from dotenv import load_dotenv


class Configuration:
    """Manages configuration and environment variables for the MCP Slackbot."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        # self.load_env()
        self.slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.slack_app_token = os.getenv("SLACK_APP_TOKEN")
        self.openai_api_base_url = os.getenv(
            "OPENAI_API_BASE_URL", "https://api.openai.com/v1/"
        )
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")

        # Ollama specific config
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the appropriate LLM API key based on the model.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If no API key is found for the selected model.
        """
        # For Ollama (local) models, no API key needed
        if self.llm_model.startswith("ollama/"):
            return None

        # For OpenRouter models
        if self.openrouter_api_key and "openrouter" in self.openai_api_base_url:
            return self.openrouter_api_key

        # For standard providers
        if "gpt" in self.llm_model.lower() and self.openai_api_key:
            return self.openai_api_key

        raise ValueError(f"No API key found for the selected model: {self.llm_model}")
