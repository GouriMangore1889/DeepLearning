import streamlit as st
from rag import RAGVectorStore
from chains import RAGChain

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 RAG Chatbot")

@st.cache_resource
def load_chain():
    vs = RAGVectorStore()
    retriever = vs.get_retriever()
    return RAGChain(retriever)

rag = load_chain()

# store chat in UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
if prompt := st.chat_input("Ask something..."):

    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.ask(prompt)
            answer = result["answer"]

            st.markdown(answer)

           
    # save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})