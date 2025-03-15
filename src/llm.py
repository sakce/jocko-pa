import logging

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ChatClient:
    """Client for communicating with LLMs via Langchain."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
    ) -> None:
        """Initialize the Langchain LLM client.

        Args:
            api_key: API key for hosted models (OpenRouter/OpenAI)
            model: Model identifier to use
            base_url: Base URL for API requests (for OpenRouter)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.agent = None

        # Initialize the appropriate chat model
        if model and model.startswith("ollama/"):
            # Format is "ollama/model_name:no_of_params"
            model_name = model.split("/", 1)[1]
            self.chat = ChatOllama(model=model_name)
            logging.info(f"Initialized Ollama with model {model_name}")
        else:
            # For OpenRouter or OpenAI models
            self.chat = ChatOpenAI(
                api_key=api_key,  # type: ignore
                model=model,
                base_url=base_url,
                temperature=0.7,
                max_completion_tokens=1500,
                streaming=False,
            )
            logging.info(f"Initialized hosted model {model} via {base_url}")
