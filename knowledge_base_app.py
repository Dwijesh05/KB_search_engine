import streamlit as st
import os
import io
import tempfile
import time
import json
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import torch
from typing import List, Dict, Tuple, Any
import re
from pathlib import Path
import base64
import pickle
import hashlib
from datetime import datetime

# Configuration
class Config:
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, efficient model
    SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"  # Lightweight summarization model
    
    # Chunking settings
    CHUNK_SIZE = 500  # Words per chunk
    CHUNK_OVERLAP = 50  # Words overlap between chunks
    
    # Vector database settings
    VECTOR_DB_PATH = "vector_db"
    METADATA_PATH = "metadata.pkl"
    
    # UI settings
    PAGE_TITLE = "Local Knowledge Base Search Engine"
    PAGE_ICON = "🔍"
    
    # File settings
    MAX_FILES = 50
    SUPPORTED_FORMATS = ["pdf", "txt"]

# Text extraction utilities
class TextExtractor:
    @staticmethod
    def extract_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF using PyMuPDF"""
        text = ""
        try:
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
        return text
    
    @staticmethod
    def extract_from_txt(file_content: bytes) -> str:
        """Extract text from TXT file"""
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_content.decode("latin-1")
            except Exception as e:
                st.error(f"Error extracting TXT: {str(e)}")
                return ""

# Text processing utilities
class TextProcessor:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = Config.CHUNK_SIZE, 
                  overlap: int = Config.CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
        return text.strip()

# Vector database handler
class VectorDB:
    def __init__(self, embedding_model: str = Config.EMBEDDING_MODEL):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        self.metadata = []
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None):
        """Add texts to the vector database"""
        if not texts:
            return
            
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy()
        
        # Initialize FAISS index if needed
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add to index
        self.index.add(embeddings_np)
        
        # Store chunks and metadata
        self.chunks.extend(texts)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{}] * len(texts))
    
    def search(self, query: str, k: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
        """Search for similar texts"""
        if self.index is None or not self.chunks:
            return [], [], []
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        query_np = query_embedding.cpu().numpy()
        
        # Search
        distances, indices = self.index.search(query_np, min(k, len(self.chunks)))
        
        # Return results
        results = []
        metadatas = []
        scores = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
                metadatas.append(self.metadata[idx])
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + distances[0][i])
                scores.append(similarity)
                
        return results, metadatas, scores
    
    def save(self, path: str = Config.VECTOR_DB_PATH):
        """Save the vector database to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save chunks and metadata
        with open(os.path.join(path, Config.METADATA_PATH), "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata
            }, f)
    
    def load(self, path: str = Config.VECTOR_DB_PATH):
        """Load the vector database from disk"""
        if not os.path.exists(path):
            return False
            
        # Load FAISS index
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # Load chunks and metadata
        metadata_path = os.path.join(path, Config.METADATA_PATH)
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data.get("chunks", [])
                self.metadata = data.get("metadata", [])
                
        return True
    
    def clear(self):
        """Clear the vector database"""
        self.index = None
        self.chunks = []
        self.metadata = []
        
        # Remove saved files
        if os.path.exists(Config.VECTOR_DB_PATH):
            for file in os.listdir(Config.VECTOR_DB_PATH):
                os.remove(os.path.join(Config.VECTOR_DB_PATH, file))

# Local LLM handler
class LocalLLM:
    def __init__(self, model_name: str = Config.SUMMARIZATION_MODEL):
        self.summarizer = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load the summarization model"""
        try:
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Using a fallback summarization method.")
            self.summarizer = None
    
    def summarize(self, text: str, max_length: int = 150) -> str:
        """Summarize text using the local model"""
        if not text.strip():
            return ""
            
        if self.summarizer is None:
            # Fallback: extract first few sentences
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '.'
        
        try:
            # Truncate text if too long
            if len(text) > 1024:
                text = text[:1024]
                
            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            return result[0]['summary_text']
        except Exception as e:
            st.error(f"Error in summarization: {str(e)}")
            # Fallback: extract first few sentences
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '.'
    
    def synthesize_answer(self, query: str, contexts: List[str]) -> str:
        """Synthesize an answer from retrieved contexts"""
        if not contexts:
            return "I couldn't find relevant information to answer your question."
        
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        
        # Create a prompt for the model
        prompt = f"""Based on the following context, please answer the question: {query}

Context:
{combined_context}

Answer:"""
        
        # For this demo, we'll use a simple approach
        # In a real implementation, you might use a more sophisticated LLM
        try:
            if self.summarizer is not None:
                # Use the summarizer to extract key information
                result = self.summarizer(
                    prompt,
                    max_length=200,
                    min_length=50,
                    do_sample=False
                )
                return result[0]['summary_text']
        except:
            pass
        
        # Fallback: extract relevant sentences
        sentences = combined_context.split('. ')
        relevant_sentences = []
        
        for sentence in sentences:
            # Simple keyword matching
            query_words = set(query.lower().split())
            sentence_words = set(sentence.lower().split())
            
            if query_words & sentence_words:  # Intersection
                relevant_sentences.append(sentence)
        
        if not relevant_sentences:
            relevant_sentences = sentences[:3]  # First few sentences as fallback
        
        return '. '.join(relevant_sentences[:3]) + '.'

# Main application
class KnowledgeBaseApp:
    def __init__(self):
        self.vector_db = VectorDB()
        self.llm = LocalLLM()
        self.documents = {}
        
        # Try to load existing database
        if self.vector_db.load():
            st.success("Loaded existing knowledge base.")
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and add to knowledge base"""
        if not uploaded_files:
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension not in Config.SUPPORTED_FORMATS:
                st.warning(f"Skipping unsupported file: {uploaded_file.name}")
                continue
            
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Extract text
            file_content = uploaded_file.read()
            
            if file_extension == "pdf":
                text = TextExtractor.extract_from_pdf(file_content)
            else:  # txt
                text = TextExtractor.extract_from_txt(file_content)
            
            if not text:
                st.warning(f"No text extracted from {uploaded_file.name}")
                continue
            
            # Clean text
            text = TextProcessor.clean_text(text)
            
            # Generate document ID
            doc_id = hashlib.md5(uploaded_file.name.encode()).hexdigest()
            
            # Store document
            self.documents[doc_id] = {
                "name": uploaded_file.name,
                "content": text,
                "summary": "",
                "uploaded_at": datetime.now().isoformat()
            }
            
            # Summarize document
            with st.spinner(f"Summarizing {uploaded_file.name}..."):
                summary = self.llm.summarize(text)
                self.documents[doc_id]["summary"] = summary
            
            # Chunk text
            chunks = TextProcessor.chunk_text(text)
            
            # Create metadata for each chunk
            metadatas = []
            for j, chunk in enumerate(chunks):
                metadatas.append({
                    "doc_id": doc_id,
                    "doc_name": uploaded_file.name,
                    "chunk_id": j,
                    "summary": summary
                })
            
            # Add to vector database
            self.vector_db.add_texts(chunks, metadatas)
            
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
        
        # Save vector database
        self.vector_db.save()
        
        status_text.text("Processing complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"Successfully processed {len(uploaded_files)} files.")
    
    def search_knowledge_base(self, query: str) -> Dict:
        """Search the knowledge base and synthesize an answer"""
        if not query:
            return {"answer": "", "sources": []}
        
        with st.spinner("Searching knowledge base..."):
            # Retrieve relevant chunks
            chunks, metadatas, scores = self.vector_db.search(query, k=5)
            
            if not chunks:
                return {
                    "answer": "I couldn't find relevant information in the knowledge base.",
                    "sources": []
                }
        
        with st.spinner("Synthesizing answer..."):
            # Synthesize answer
            answer = self.llm.synthesize_answer(query, chunks)
            
            # Prepare sources
            sources = []
            seen_docs = set()
            
            for chunk, metadata, score in zip(chunks, metadatas, scores):
                doc_id = metadata.get("doc_id")
                if doc_id and doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    doc_info = self.documents.get(doc_id, {})
                    sources.append({
                        "name": doc_info.get("name", "Unknown"),
                        "summary": doc_info.get("summary", ""),
                        "relevance": score
                    })
        
        return {"answer": answer, "sources": sources}
    
    def clear_knowledge_base(self):
        """Clear the knowledge base"""
        self.vector_db.clear()
        self.documents = {}
        st.success("Knowledge base cleared.")
    
    def run(self):
        """Run the Streamlit app"""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for dark mode
        st.markdown("""
        <style>
        :root {
            --primary-color: #6366f1;
            --background-color: #0f172a;
            --secondary-background-color: #1e293b;
            --text-color: #e2e8f0;
            --secondary-text-color: #94a3b8;
        }
        
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .stSidebar {
            background-color: var(--secondary-background-color);
        }
        
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #4f46e5;
        }
        
        .stTextInput>div>div>input {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid #334155;
        }
        
        .stTextArea>div>div>textarea {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid #334155;
        }
        
        .stFileUploader>div>div>div {
            background-color: var(--secondary-background-color);
            border: 1px dashed #334155;
        }
        
        .source-card {
            background-color: var(--secondary-background-color);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid var(--primary-color);
        }
        
        .answer-box {
            background-color: var(--secondary-background-color);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #334155;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color);
        }
        
        p, li, span {
            color: var(--text-color);
        }
        
        .highlight {
            background-color: rgba(99, 102, 241, 0.2);
            padding: 2px 4px;
            border-radius: 3px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("🔍 Local Knowledge Base Search Engine")
        st.markdown("Upload documents, search, and get answers - all offline!")
        
        # Sidebar
        with st.sidebar:
            st.header("Knowledge Base Management")
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload PDF or TXT files",
                type=Config.SUPPORTED_FORMATS,
                accept_multiple_files=True,
                help=f"Upload up to {Config.MAX_FILES} files"
            )
            
            if uploaded_files:
                if st.button("Process Files", type="primary"):
                    if len(uploaded_files) > Config.MAX_FILES:
                        st.error(f"Please upload no more than {Config.MAX_FILES} files.")
                    else:
                        self.process_uploaded_files(uploaded_files)
            
            st.divider()
            
            # Knowledge base info
            if self.documents:
                st.subheader("Knowledge Base Info")
                st.write(f"Documents: {len(self.documents)}")
                st.write(f"Chunks: {len(self.vector_db.chunks)}")
                
                # Document list
                with st.expander("View Documents"):
                    for doc_id, doc_info in self.documents.items():
                        st.write(f"**{doc_info['name']}**")
                        st.caption(f"Summary: {doc_info['summary'][:100]}...")
                        st.caption(f"Uploaded: {doc_info['uploaded_at']}")
                        st.divider()
            
            st.divider()
            
            # Clear button
            if st.button("Clear Knowledge Base", type="secondary"):
                if st.session_state.get('confirm_clear', False):
                    self.clear_knowledge_base()
                    st.session_state.confirm_clear = False
                else:
                    st.session_state.confirm_clear = True
                    st.warning("Click again to confirm clearing the knowledge base.")
        
        # Main content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.header("Search")
            query = st.text_input("Enter your question:", placeholder="What would you like to know?")
            
            if st.button("Search", type="primary") and query:
                results = self.search_knowledge_base(query)
                
                # Display answer
                if results["answer"]:
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.subheader("Answer")
                    st.write(results["answer"])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display sources
                if results["sources"]:
                    st.subheader("Sources")
                    for source in results["sources"]:
                        st.markdown(f'''
                        <div class="source-card">
                            <h4>{source["name"]}</h4>
                            <p>{source["summary"]}</p>
                            <p><small>Relevance: {source["relevance"]:.2f}</small></p>
                        </div>
                        ''', unsafe_allow_html=True)
        
        with col2:
            st.header("Quick Actions")
            
            # Download summaries
            if self.documents:
                summaries = {doc_info["name"]: doc_info["summary"] 
                            for doc_info in self.documents.values()}
                
                if st.button("Download All Summaries"):
                    summaries_json = json.dumps(summaries, indent=2)
                    b64 = base64.b64encode(summaries_json.encode()).decode()
                    href = f'<a href="data:file/json;base64,{b64}" download="summaries.json">Download JSON</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            # Statistics
            if self.documents:
                st.subheader("Statistics")
                total_chars = sum(len(doc["content"]) for doc in self.documents.values())
                st.metric("Total Characters", f"{total_chars:,}")
                st.metric("Average Chunks per Doc", f"{len(self.vector_db.chunks) / len(self.documents):.1f}")

# Run the app
if __name__ == "__main__":
    app = KnowledgeBaseApp()
    app.run()
