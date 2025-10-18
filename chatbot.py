from state import stateChatbot
from function import chat_node, tools_node
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import tools_condition

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

checkPointer = SqliteSaver(conn=conn)
graph = StateGraph(stateChatbot)

graph.add_node("chat", chat_node)
graph.add_node("tools",tools_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat",tools_condition)
graph.add_edge("tools","chat")

workflow = graph.compile(checkpointer=checkPointer)

def retrieve():
    thread = set()
    for i in checkPointer.list(None):
        thread.add(i.config['configurable']["thread_id"])
    return list(thread)

