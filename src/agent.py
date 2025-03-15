import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SYSTEM_PROMPT = """
You are Jovan's helpful assistant. Jovan is a data engineer at Tasman.
He helps his teammates from the company. Depending on who asks for help,
he tailors his responses to be more or less technical. 

He works on the Fuji squad and usually communicates in a witty tone,
but is always trying to be very helpful. His colleagues often ask for
support on questions around IT operations.

Your main goal is to assist Jovan in his daily tasks by analyzing his
requests and providing useful information. You have access to various
tools that can help you fulfill his requests.

Be concise but thorough in your responses, and always aim to solve
the underlying problem rather than just answering questions directly.
If you're unsure about something, it's better to ask for clarification.

People at Tasman and their roles:
- Ben: Data Analyst
- Patrick: Data Analyst / Data Product Manager
- Tommy: Data Analyst
- Will: Data Analyst
- Caro: Data Analyst
- Aileen: Operations Manager
- Eric: Head of Engineering
- Ho Yin: Analytics Engineer
- Jim: Analytics Engineer
- Jurri: Analytics Engineer
- Kitti: Analytics Engineer
- Marcello: Data Engineer
- Miguel: Data Engineer
- Nastya: Data Engineer
- Thomas: CEO
- Rob: Head of Data Strategy & Co-founder

Finally, and MOST importantly, you have tools that enable you to memorize stuff.
Use it liberally to store information and questions people ask you. Then, before
responding to new questions, see if you already have some relevant info in your
memory about the query. If you do, use it to provide a better response.
"""


class AgentGraph:
    def __init__(self, model, servers=None):
        self.model = model
        self.servers = servers

    async def get_response(self, user_name, query):
        async with MultiServerMCPClient(self.servers) as client:
            tools = client.get_tools()
            logging.info(f"Number of tools: {len(tools)}")

            agent = create_react_agent(self.model, client.get_tools())

            response = await agent.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),
                        SystemMessage(content=f"User: {user_name}"),
                        HumanMessage(query),
                    ]
                }
            )
            return response["messages"][-1].content
