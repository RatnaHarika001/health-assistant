from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages,BaseMessage
from langgraph.graph.message import AnyMessage
from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

class State(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


def make_workflow_graph():

    graph = StateGraph(State)

    def call_model(state):
        return {"messages":[llm.invoke(state['messages'])]}
    
    graph.add_node('agent',call_model)
    graph.add_edge(START,'agent')
    graph.add_edge('agent',END)

    agent = graph.compile()

    return agent

agent = make_workflow_graph()