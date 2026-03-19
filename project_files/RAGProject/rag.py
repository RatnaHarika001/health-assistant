from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages,AnyMessage
from typing import Annotated,TypedDict,List
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import Tool
from langgraph.prebuilt import create_react_agent
import streamlit as st

import os
from dotenv import load_dotenv
os.environ["OPENAI_API_KEY"]="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"

llm = ChatOpenAI(model="gpt-4o-mini")

def make_retriever_tool_from_pdf(file,name,desc):
    docs = PyPDFLoader(file_path=file).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50).split_documents(documents=docs)
    vs = FAISS.from_documents(documents=chunks,embedding=OpenAIEmbeddings())
    retriever = vs.as_retriever()

    def tool_func(query:str)->str:
        print(f"Using Tool {name}")
        result = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in result)
    
    return Tool(name=name,func=tool_func,description=desc)

class State(TypedDict):
    messages : Annotated[List[AnyMessage],add_messages]


# ---------------------------
@st.cache_resource
def load_agent():
    llm = ChatOpenAI(model="gpt-4o-mini")

    tool = make_retriever_tool_from_pdf(
        "Hypertension_1.pdf",
        "hypertension_guide",
        "Use for blood pressure questions"
    )

    return create_react_agent(llm, tools=[tool])

agent = load_agent()

st.set_page_config(page_title="Health Assistant", page_icon="🏥")

st.title("🏥 Smart Health Assistant 🤖")
st.caption("Ask health-related questions from medical documents")

st.warning("⚠️ This is for educational purposes only. Consult a doctor for medical advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask your question..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.invoke({
                "messages": [
                    ("system", 
                                "You are a medical assistant. ONLY answer questions which have answers in the pdf of tools "
                                "If the answer to the question is not present in pdf of tools, say: 'I can only answer medical-related questions.'"),
                    *[(m["role"], m["content"]) for m in st.session_state.messages]
                ]
            })

            answer = response["messages"][-1].content
            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})





