import os
from typing import List, Dict, Tuple
import autogen
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from dotenv import load_dotenv
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

class RAGAgent:
    def __init__(self, documents_path: str, cache_dir: str = "vector_cache"):
        self.documents_path = documents_path
        self.cache_dir = cache_dir
        # Use a more stable embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize vector store
        self.initialize_vectorstore()
        
        # Configure AutoGen agents
        self.config_list = [
            {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ]
        
        # Create retriever agent
        self.retriever_agent = autogen.AssistantAgent(
            name="retriever",
            llm_config={"config_list": self.config_list},
            system_message="You are a retrieval expert responsible for finding relevant information from the knowledge base."
        )
        
        # Create answer agent
        self.answer_agent = autogen.AssistantAgent(
            name="answerer",
            llm_config={"config_list": self.config_list},
            system_message="You are an expert in analyzing transactions and identifying potential anomalies."
        )
        
        # Create user proxy
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={"work_dir": "workspace"},
            llm_config={"config_list": self.config_list},
        )

    def initialize_vectorstore(self):
        """Initialize vector store with caching mechanism"""
        persist_directory = os.path.join(self.cache_dir, "chroma_db")
        
        # Check if vector store exists
        if os.path.exists(persist_directory):
            print("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            return
            
        print("Initializing new vector store...")
        if not os.path.exists(self.documents_path):
            raise FileNotFoundError(f"Document path {self.documents_path} does not exist")
            
        # Load all documents using DirectoryLoader
        malicious_loader = DirectoryLoader(
            os.path.join(self.documents_path, "malicious"),
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        normal_loader = DirectoryLoader(
            os.path.join(self.documents_path, "normal"),
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        
        # Load documents
        malicious_docs = malicious_loader.load()
        normal_docs = normal_loader.load()
        
        # Add labels to documents
        for doc in malicious_docs:
            doc.metadata["type"] = "malicious"
        for doc in normal_docs:
            doc.metadata["type"] = "normal"
            
        all_docs = malicious_docs + normal_docs
        
        # Use more granular text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size
            chunk_overlap=50,  # Reduced overlap
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        print("Splitting documents...")
        texts = text_splitter.split_documents(all_docs)
        
        # Create vector store
        print("Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Persist the vector store
        print("Persisting vector store...")
        self.vectorstore.persist()

    def retrieve_relevant_docs(self, query: str, k: int = 5) -> Dict[str, List[str]]:
        """Retrieve relevant documents using MMR algorithm for diverse retrieval"""
        # Use MMR algorithm for retrieval
        docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=20  # Get more candidate documents
        )
        
        # Group by type
        malicious_docs = []
        normal_docs = []
        
        for doc in docs:
            if doc.metadata.get("type") == "malicious":
                malicious_docs.append(doc.page_content)
            else:
                normal_docs.append(doc.page_content)
        
        # Return balanced results
        return {
            "malicious": malicious_docs,
            "normal": normal_docs
        }

    def process_query(self, query: str) -> str:
        """Process user query"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(query)
        
        # Build context
        context = {
            "malicious": "\n".join(relevant_docs["malicious"]),
            "normal": "\n".join(relevant_docs["normal"])
        }
        
        # Build prompt
        prompt = f"""Based on the following context, analyze the transaction:

Malicious Transaction Context:
{context['malicious']}

Normal Transaction Context:
{context['normal']}

Query: {query}

Please analyze if this transaction is anomalous. If it is anomalous, please specify what type of anomaly it belongs to.
"""
        
        # Use AutoGen for conversation
        self.user_proxy.initiate_chat(
            self.retriever_agent,
            message=prompt
        )
        
        # Get response
        response = self.user_proxy.last_message()["content"]
        return response

    def analyze_transaction(self, transaction_data: Dict) -> Tuple[bool, str]:
        """
        Analyze transaction data and determine if it's an anomalous transaction
        
        Args:
            transaction_data: Transaction data dictionary
            
        Returns:
            Tuple[bool, str]: (Is it anomalous, Anomaly type)
        """
        # Convert transaction data to feature vector
        features = self._extract_features(transaction_data)
        
        # Use anomaly detection model for prediction
        is_anomaly = self.anomaly_detector.predict([features])[0] == -1
        
        if is_anomaly:
            # Use RAG system to analyze anomaly type
            anomaly_type = self._classify_anomaly(transaction_data)
            return True, anomaly_type
        return False, "normal"

    def _extract_features(self, transaction_data: Dict) -> List[float]:
        """Extract transaction features"""
        # This needs to be implemented based on the actual structure of transaction data
        # Example features: transaction amount, gas fee, timestamp, etc.
        features = []
        # TODO: Implement specific feature extraction logic
        return features

    def _classify_anomaly(self, transaction_data: Dict) -> str:
        """Classify anomalous transaction"""
        # Build query
        query = f"Analyze the following transaction data and determine what type of anomaly it belongs to: {json.dumps(transaction_data)}"
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(query)
        context = "\n".join(relevant_docs["malicious"])
        
        # Build prompt
        prompt = f"""Based on the following context, analyze the transaction anomaly type:

Context:
{context}

Transaction Data:
{json.dumps(transaction_data, indent=2)}

Please determine what type of anomaly this transaction belongs to (hack/exploit/scam/phishing/malware).
"""
        
        # Use AutoGen for conversation
        self.user_proxy.initiate_chat(
            self.retriever_agent,
            message=prompt
        )
        
        # Get response
        response = self.user_proxy.last_message()["content"]
        return response

def main():
    # Usage example
    rag_agent = RAGAgent(documents_path="documents")
    query = "Analyze if this transaction is anomalous: 0x123... transferring 1.5 ETH to 0x456..."
    response = rag_agent.process_query(query)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main() 
