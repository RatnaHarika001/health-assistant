from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages,BaseMessage
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode

class State(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

llm = ChatOpenAI(model="gpt-4o-mini")

def make_workflow_graph():

    graph = StateGraph(State)

    def call_model(state):
        return {"messages":llm.invoke(state['messages'])}
    
    graph.add_node('agent',call_model)
    graph.add_edge(START,'agent')
    graph.add_edge('agent',END)

    agent = graph.compile()

    return agent

agent = make_workflow_graph()