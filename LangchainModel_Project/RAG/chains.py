import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class RAGChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.3
        )
        self.chat_history = []   # stores messages

    def ask(self, question: str):

        # 🔹 Clean query
        question = question.strip()

        if len(question) < 3:
            return {
                "answer": "Please ask a meaningful question.",
                "sources": []
            }
        question_lower = question.lower()

        # 🔹 Greeting handling (before retrieval)
        if question_lower in ["hi", "hello", "hey"]:
            return {
                "answer": "Hello! How can I help you today?",
                "sources": []
            }

        if "how are you" in question_lower:
            return {
                "answer": "I'm doing well! How can I assist you?",
                "sources": []
            }
        docs = self.retriever.invoke(question)

        # filter weak docs (basic logic)
        filtered_docs = [doc for doc in docs if len(doc.page_content) > 50]

        context = "\n\n".join(doc.page_content for doc in filtered_docs)
        sources = list(set(doc.metadata.get("source", "unknown") for doc in filtered_docs))

        # 🔹 Build messages (ChatGPT style)
        messages = []

        # system role
        messages.append({
            "role": "system",
            "content": """You are a helpful assistant.

Rules:
- Answer only from provided context
- If not found, say "I don't know"
- Keep answers clear and short
"""
        })

        # add past conversation
        for msg in self.chat_history:
            messages.append(msg)

        # add current question with context
        messages.append({
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}
"""
        })

        # 🔹 Get response
        response = self.llm.invoke(messages)
        answer = response.content

        # 🔹 Save conversation
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
        "answer": answer
         }