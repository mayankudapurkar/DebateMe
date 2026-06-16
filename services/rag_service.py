from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
import os


class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.search_tool = TavilySearchResults(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=5
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.vectorstore = None

    def fetch_and_index(self, topic: str):
        """Fetch articles on topic via Tavily and index into FAISS"""
        pro_results = self.search_tool.invoke(f"arguments in favor of {topic}")
        con_results = self.search_tool.invoke(f"arguments against {topic}")

        all_docs = []
        for result in pro_results + con_results:
            if isinstance(result, dict):
                content = result.get("content", "")
                url = result.get("url", "")
                if content:
                    all_docs.append(Document(
                        page_content=content,
                        metadata={"source": url, "topic": topic}
                    ))

        if not all_docs:
            raise ValueError(f"No articles found for topic: {topic}")

        chunks = self.splitter.split_documents(all_docs)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    def retrieve_counter_evidence(self, claim: str, k: int = 4) -> list[str]:
        """Given a user's claim, retrieve evidence that counters it"""
        if not self.vectorstore:
            raise ValueError("No documents indexed. Call fetch_and_index first.")

        query = f"evidence against: {claim}"
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]