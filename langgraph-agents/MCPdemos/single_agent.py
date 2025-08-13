import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

# Load environment variables from .env file
# https://github.com/keertipurswani/MachineLearningTutorials/blob/main/MCP/main.py
load_dotenv()

async def run_single_agent():
    #https://github.com/langchain-ai/langchain-mcp-adapters?tab=readme-ov-file
    client = MultiServerMCPClient(
        {
            "Bright Data": {
                #MCP server + WebScrapper | https://github.com/brightdata/brightdata-mcp
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": "<insert-your-api-token-here>",
                    "RATE_LIMIT": "<optional if you want to change rate limit format: limit/time+unit, e.g., 100/1h, 50/30m, 10/5s>",
                    "WEB_UNLOCKER_ZONE": "<optional if you want to override the web unlocker zone name - default is mcp_unlocker>",
                    "BROWSER_ZONE": "<optional if you want to override the browser zone name - defaults is mcp_browser>",
                    "PRO_MODE": "<optional boolean, defaults to false. Set to true to expose all tools including browser and web_data_* tools>"
                },
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    # agent = create_react_agent("openai:gpt-4.1", tools)
    model = init_chat_model(model_name="openai:gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))
    agent = create_react_agent(
        model = model,
        tools = tools,
        prompt = "You are a helpful assistant that can use various tools to answer questions."
    )
    agent_response = await agent.ainvoke({"messages": "who won the 2023 world series?"})
    print(agent_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(run_single_agent())
    # Get the OpenAI API key from environment variables
    #openai_api_key = os.getenv("OPENAI_API_KEY")