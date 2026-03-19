from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph.message import BaseMessage,add_messages
from langgraph.graph import START,StateGraph,END
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


llm = ChatOpenAI(model="gpt-4o-mini")

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

