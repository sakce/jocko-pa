import asyncio
import logging
import os

from langchain_core.messages import AIMessage, HumanMessage
from posthog import Posthog
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from src.agent import AgentGraph
from src.config import Configuration
from src.llm import ChatClient

posthog = Posthog(os.getenv("POSTHOG_KEY"), host="https://eu.i.posthog.com")
THREAD_CONTEXT_MESSAGES_LIMIT = 7


class SlackAgent:
    """Manages the Slack bot integration with MCP servers."""

    def __init__(
        self,
        config: Configuration,
        tools=None,
    ) -> None:
        self.config = config

        self.app = AsyncApp(token=self.config.slack_bot_token)
        self.socket_mode_handler = AsyncSocketModeHandler(
            self.app, self.config.slack_app_token
        )
        self.client = AsyncWebClient(token=self.config.slack_bot_token)

        self.model = ChatClient(
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
            base_url=self.config.openai_api_base_url,
        ).chat

        self.tools = tools
        self.agent = AgentGraph(self.model, self.tools)

        self.app.message()(self.handle_message)
        self.app.event("app_mention")(self.handle_mention)
        self.app.event("assistant_thread_started")(self.handle_assistant_thread_started)
        self.app.event("app_home_opened")(self.handle_home_opened)

    async def initialize_bot_info(self) -> None:
        """Get the bot's ID and other info."""
        try:
            auth_info = await self.client.auth_test()
            self.bot_id = auth_info["user_id"]
            logging.info(f"Bot initialized with ID: {self.bot_id}")
        except Exception as e:
            logging.error(f"Failed to get bot info: {e}")
            self.bot_id = None

    async def handle_home_opened(self, event, client):
        """Handle when a user opens the App Home tab."""
        user_id = event["user"]

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Jocko says hi!"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "I'm an AI assistant with access to tools and resources "
                        "through the Model Context Protocol."
                    ),
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Available Tools:*"},
            },
        ]

        for tool in self.tools:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"• *{tool.name}*: {tool.description}",
                    },
                }
            )

        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*How to Use:*\n"
                        "1. Send me a direct message or open my agent thread form the top-right corner.\n"
                        "2. Ask me to check something out from Monday.com"
                    ),
                },
            }
        )

        try:
            await client.views_publish(
                user_id=user_id, view={"type": "home", "blocks": blocks}
            )
        except Exception as e:
            logging.error(f"Error publishing home view: {e}")

    async def handle_message(self, message, say):
        """Handle direct messages to the bot."""
        channel = message.get("channel")
        user_id = message.get("user")
        thread_ts = message.get("thread_ts", message.get("ts"))

        try:
            if message.get("channel_type") == "im" and not message.get("subtype"):
                await self.app.client.assistant_threads_setStatus(
                    channel_id=channel, thread_ts=thread_ts, status="is thinking.."
                )
                user_id = message.get("user")
                posthog.capture(user_id, "user_sent_message")
                user_info = await self.client.users_info(user=user_id)
                user_first_name = user_info["user"]["profile"]["first_name"]

                # Fetch recent messages in the thread
                thread_history = []
                if thread_ts:
                    conversation_history = await self.client.conversations_replies(
                        channel=channel,
                        ts=thread_ts,
                        limit=THREAD_CONTEXT_MESSAGES_LIMIT,
                    )
                    if conversation_history and "messages" in conversation_history:
                        messages = conversation_history["messages"]
                        for msg in messages:
                            if msg.get("bot_id"):
                                thread_history.append(
                                    AIMessage(content=msg.get("text", ""))
                                )
                            else:
                                thread_history.append(
                                    HumanMessage(content=msg.get("text", ""))
                                )

                # If no thread history, just add the current message
                if not thread_history:
                    thread_history.append(HumanMessage(content=message.get("text", "")))

                response = await self.agent.get_response(
                    user_name=user_first_name, messages=thread_history
                )
                await say(response)

        except Exception as e:
            logging.error(f"Error handling message: {e}")
            await say("Sorry, Jovan's messed something up again. Please shout at him.")

    async def handle_mention(self, event, say):
        """Handle mentions of the bot in channels."""
        channel = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "")
        ts = event.get("ts")
        thread_ts = event.get("thread_ts", ts)  # Use ts as thread_ts if not in a thread

        try:
            await self.app.client.assistant_threads_setStatus(
                channel_id=channel, thread_ts=thread_ts, status="is thinking.."
            )

            posthog.capture(user_id, "user_mentioned_bot")
            user_info = await self.client.users_info(user=user_id)
            user_first_name = user_info["user"]["profile"]["first_name"]

            # Remove the bot mention from the text
            cleaned_text = text.replace(f"<@{self.bot_id}>", "").strip()

            # Fetch recent messages in the thread if it exists
            thread_history = []
            if thread_ts:
                conversation_history = await self.client.conversations_replies(
                    channel=channel,
                    ts=thread_ts,
                    limit=THREAD_CONTEXT_MESSAGES_LIMIT,
                )
                if conversation_history and "messages" in conversation_history:
                    messages = conversation_history["messages"]
                    for msg in messages:
                        if msg.get("bot_id"):
                            thread_history.append(
                                AIMessage(content=msg.get("text", ""))
                            )
                        else:
                            # Clean mentions from the text
                            msg_text = msg.get("text", "")
                            if f"<@{self.bot_id}>" in msg_text:
                                msg_text = msg_text.replace(
                                    f"<@{self.bot_id}>", ""
                                ).strip()
                            thread_history.append(HumanMessage(content=msg_text))

            # If no thread history, just add the current message
            if not thread_history:
                thread_history.append(HumanMessage(content=cleaned_text))

            response = await self.agent.get_response(
                user_name=user_first_name, messages=thread_history
            )

            # Send response in the thread
            await self.client.chat_postMessage(
                channel=channel, text=response, thread_ts=thread_ts
            )

        except Exception as e:
            logging.error(f"Error handling mention: {e}")
            await self.client.chat_postMessage(
                channel=channel,
                text="Sorry, I encountered an error while processing your request.",
                thread_ts=thread_ts,
            )

    async def handle_assistant_thread_started(self, event):
        """Handle the assistant_thread_started event."""
        user_id = event.get("assistant_thread").get("user_id")
        posthog.capture(user_id, "assistant_thread_started")
        logging.info(f"Assistant thread started with user: {user_id}")

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
