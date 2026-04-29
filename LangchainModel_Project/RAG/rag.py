import os
from typing import List
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

class RAGVectorStore:
    def __init__(self, documents_dir="documents"):
        self.documents_dir = documents_dir

        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL"),
            model_kwargs={'device': 'cpu'}
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.vectorstore = None

    def load_documents(self):
        documents = []

        for filename in os.listdir(self.documents_dir):
            filepath = os.path.join(self.documents_dir, filename)

            # TXT FILE
            if filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    documents.append(
                        Document(
                            page_content=f.read(),
                            metadata={"source": filename}
                        )
                    )

            # PDF FILE
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                pdf_docs = loader.load()

                for doc in pdf_docs:
                    doc.metadata["source"] = filename
                    documents.append(doc)

        if not documents:
            raise ValueError("No documents found")

        return documents

    def get_retriever(self):
        if self.vectorstore is None:
            docs = self.load_documents()
            splits = self.splitter.split_documents(docs)
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}   # try 3–5
        )