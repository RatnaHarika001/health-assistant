from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AnyMessage   # ✅ moved here
from langgraph.graph import add_messages         # ✅ moved here
from typing import Annotated, TypedDict, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver   # ✅ NEW (important)
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OpenAI API key not found. Please set it in Streamlit secrets.")
    st.stop()

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pdf_path = os.path.join(BASE_DIR, "project_files", "RAGProject", "Hypertension_1.pdf")



llm = ChatOpenAI(model="gpt-4o-mini")

def make_retriever_tool_from_pdf(file,name,desc):
    docs = PyPDFLoader(file_path=file).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(documents=docs)
    vs = FAISS.from_documents(documents=chunks,embedding=OpenAIEmbeddings())
    retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

    def tool_func(query: str) -> str:
        result = retriever.get_relevant_documents(query)

        if not result:
            return "No relevant Info"
    
        print("Query:", query)
        print("Retrieved docs:", len(result))
    
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
                                """You are a medical assistant. Use the provided tool information to answer.
                                If the tool provides useful information, answer clearly.If no useful information is found, say:
                                "I don’t have enough information from the documents.""""),
                    *[(m["role"], m["content"]) for m in st.session_state.messages]
                ]
            })

            answer = response["messages"][-1].content
            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})



