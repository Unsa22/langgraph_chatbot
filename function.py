from model import model
from state import stateChatbot
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchResults
from test import chain

search= DuckDuckGoSearchResults(output_format="list")
tools= [search]
model_with_tools= model.bind_tools(tools)

tools_node= ToolNode(tools)


def chat_node(state:stateChatbot):
    userMessage=state['messages'][-1].content
    response_text=chain.invoke(userMessage)
    response_message={"messages":[response_text]}
    return response_message