import asyncio
import logging

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient
from src.agent import AgentGraph
from src.config import Configuration
from src.llm import ChatClient


class SlackAgent:
    """Manages the Slack bot integration with MCP servers."""

    def __init__(
        self,
    ) -> None:
        self.config = Configuration()

        self.app = AsyncApp(token=self.config.slack_bot_token)
        self.socket_mode_handler = AsyncSocketModeHandler(
            self.app, self.config.slack_app_token
        )
        self.client = AsyncWebClient(token=self.config.slack_bot_token)

        self.servers = self.config.load_config("servers_config.json")

        self.model = ChatClient(
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
            base_url=self.config.openai_api_base_url,
        ).chat

        self.app.message()(self.handle_message)

    async def initialize_bot_info(self) -> None:
        """Get the bot's ID and other info."""
        try:
            auth_info = await self.client.auth_test()
            self.bot_id = auth_info["user_id"]
            logging.info(f"Bot initialized with ID: {self.bot_id}")
        except Exception as e:
            logging.error(f"Failed to get bot info: {e}")
            self.bot_id = None

    async def handle_message(self, message, say):
        """Handle direct messages to the bot."""
        if message.get("channel_type") == "im" and not message.get("subtype"):
            user_id = message.get("user")
            user_info = await self.client.users_info(user=user_id)
            user_first_name = user_info["user"]["profile"]["first_name"]
            agent = AgentGraph(self.model, self.servers)
            response = await agent.get_response(
                user_name=user_first_name, query=message.get("text")
            )
            await say(response)

    async def start(self) -> None:
        """Start the Slack bot."""
        await self.initialize_bot_info()
        logging.info("Starting Slack bot...")
        asyncio.create_task(self.socket_mode_handler.start_async())
        logging.info("Slack bot started and waiting for messages")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, "socket_mode_handler"):
                await self.socket_mode_handler.close_async()
            logging.info("Slack socket mode handler closed")
        except Exception as e:
            logging.error(f"Error closing socket mode handler: {e}")
