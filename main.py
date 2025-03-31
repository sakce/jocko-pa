import asyncio
import logging

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import Configuration
from src.slack import SlackAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def main() -> None:
    """Initialize and run the Slack bot."""
    config = Configuration()

    if not config.slack_bot_token or not config.slack_app_token:
        raise ValueError(
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in environment variables"
        )

    servers = config.load_config("servers_config.json")
    async with MultiServerMCPClient(servers) as client:
        tools = client.get_tools()
        logging.info(f"Number of tools: {len(tools)}")

        slack_bot = SlackAgent(config, tools)

        try:
            await slack_bot.start()
            # Keep the main task alive until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            await slack_bot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
