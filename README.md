# Evil Jovan Slack Bot

## Setup

- Create a `servers_config.json` file in the root of the repository. Fill the JSON with MCP server configurations as defined [here](https://github.com/langchain-ai/langchain-mcp-adapters?tab=readme-ov-file#using-with-langgraph-api-server).
- Copy the `.env.example` file to `.env` and fill in the values.
- Depending on which LLM you want to run, you might need to turn on Ollama.
- Run: `just run`