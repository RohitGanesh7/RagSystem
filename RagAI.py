import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import pickle
from datetime import datetime

class PDFRAGSystem:
    def __init__(self, openai_api_key: str, use_streamlit: bool = False):
        """
        Initialize the PDF RAG System
        
        Args:
            openai_api_key: Your OpenAI API key
            use_streamlit: Whether to use Streamlit for output
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.use_streamlit = use_streamlit
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vector_store = None
        self.qa_chain = None
        self.documents_loaded = []
        
        # Create custom prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions based on provided PDF documents.

Context from PDFs:
{context}

Question: {question}

Instructions:
1. Answer the question using ONLY the information provided in the context
2. If the answer is not in the context, say "I cannot find this information in the provided documents"
3. Be specific and cite relevant details from the documents
4. If you're unsure, say so rather than guessing

Answer:"""
        )
    
    def _log(self, message: str, level: str = "info"):
        """Log message to appropriate output (Streamlit or console)"""
        if self.use_streamlit:
            if level == "success":
                st.success(message)
            elif level == "error":
                st.error(message)
            elif level == "warning":
                st.warning(message)
            else:
                st.info(message)
        else:
            print(message)
    
    def load_pdf(self, pdf_path: str) -> List[Dict]:
        """Load and process a single PDF file"""
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata['source_file'] = os.path.basename(pdf_path)
                chunk.metadata['loaded_at'] = datetime.now().isoformat()
            
            self._log(f"✅ Loaded {len(chunks)} chunks from {os.path.basename(pdf_path)}", "success")
            return chunks
            
        except Exception as e:
            self._log(f"❌ Error loading {pdf_path}: {str(e)}", "error")
            return []
    
    def load_multiple_pdfs(self, pdf_directory: str) -> List[Dict]:
        """Load multiple PDF files from a directory"""
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise ValueError(f"Directory {pdf_directory} does not exist")
        
        all_chunks = []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_directory}")
        
        self._log(f"📁 Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            chunks = self.load_pdf(str(pdf_file))
            all_chunks.extend(chunks)
            self.documents_loaded.append(str(pdf_file))
        
        self._log(f"📚 Total chunks loaded: {len(all_chunks)}", "success")
        return all_chunks
    
    def create_vector_store(self, chunks: List[Dict]) -> None:
        """Create FAISS vector store from document chunks"""
        if not chunks:
            raise ValueError("No chunks provided to create vector store")
        
        self._log("🔄 Creating embeddings and vector store...")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        self._log("✅ Vector store created successfully", "success")
    
    def setup_qa_chain(self) -> None:
        """Setup the question-answering chain"""
        if self.vector_store is None:
            raise ValueError("Vector store not created. Load documents first.")
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
        
        self._log("✅ QA chain setup complete", "success")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer from the PDFs"""
        if self.qa_chain is None:
            raise ValueError("QA chain not setup. Initialize the system first.")
        
        # Don't log question processing in Streamlit (it's redundant)
        if not self.use_streamlit:
            self._log(f"🤔 Processing question: {question}")
        
        # Get answer
        result = self.qa_chain({"query": question})
        
        # Extract source information
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "source_file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            })
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": sources,
            "confidence": "High" if len(result["source_documents"]) >= 3 else "Medium"
        }
    
    def save_vector_store(self, save_path: str) -> None:
        """Save the vector store to disk for faster loading later"""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        # Save FAISS index
        self.vector_store.save_local(save_path)
        
        # Save metadata
        metadata = {
            "documents_loaded": self.documents_loaded,
            "created_at": datetime.now().isoformat(),
            "total_chunks": self.vector_store.index.ntotal
        }
        
        with open(f"{save_path}/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        self._log(f"💾 Vector store saved to {save_path}", "success")
    
    def load_vector_store(self, load_path: str) -> None:
        """Load a previously saved vector store"""
        # Load FAISS index
        self.vector_store = FAISS.load_local(
            load_path, 
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load metadata
        try:
            with open(f"{load_path}/metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
                self.documents_loaded = metadata.get("documents_loaded", [])
                self._log(f"📚 Loaded vector store with {metadata.get('total_chunks', 0)} chunks", "success")
        except FileNotFoundError:
            self._log("⚠️ Metadata file not found, continuing without it", "warning")
        
        self._log("✅ Vector store loaded successfully", "success")
    
    def get_similar_documents(self, question: str, k: int = 5) -> List[Dict]:
        """Get similar documents for a question without generating an answer"""
        if self.vector_store is None:
            raise ValueError("Vector store not created")
        
        # Search for similar documents
        docs = self.vector_store.similarity_search(question, k=k)
        
        similar_docs = []
        for doc in docs:
            similar_docs.append({
                "content": doc.page_content,
                "source_file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "similarity_score": "High"
            })
        
        return similar_docs
    
    def initialize_from_directory(self, pdf_directory: str, save_path: str = None) -> None:
        """Complete initialization from a directory of PDFs"""
        self._log("🚀 Initializing PDF RAG System...")
        
        # Load all PDFs
        chunks = self.load_multiple_pdfs(pdf_directory)
        
        # Create vector store
        self.create_vector_store(chunks)
        
        # Setup QA chain
        self.setup_qa_chain()
        
        # Save if requested
        if save_path:
            self.save_vector_store(save_path)
        
        self._log("🎉 PDF RAG System ready for questions!", "success")
    
    def initialize_from_saved(self, load_path: str) -> None:
        """Initialize from a previously saved vector store"""
        self._log("📂 Loading saved vector store...")
        
        # Load vector store
        self.load_vector_store(load_path)
        
        # Setup QA chain
        self.setup_qa_chain()
        
        self._log("🎉 PDF RAG System ready for questions!", "success")

# Streamlit Web Interface
def create_streamlit_app():
    """Create a Streamlit web interface for the PDF RAG system"""
    
    st.set_page_config(
        page_title="PDF Question Answering System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Question Answering System")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    # Initialize with hardcoded values
    api_key = TESTING_CONFIG["API_KEY"]
    pdf_dir = TESTING_CONFIG["PDF_DIRECTORY"]
    save_path = TESTING_CONFIG["SAVE_PATH"]
    
    # Sidebar for all controls
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check if system is already initialized
        if not st.session_state.documents_loaded:
            if st.button("🚀 Initialize System", type="primary"):
                with st.spinner("Initializing PDF RAG System..."):
                    try:
                        # Initialize RAG system
                        rag_system = PDFRAGSystem(api_key, use_streamlit=True)
                        
                        # Check if saved index exists
                        if save_path and os.path.exists(save_path):
                            st.info("📂 Loading existing index...")
                            rag_system.initialize_from_saved(save_path)
                        else:
                            st.info("🆕 Creating new index from PDFs...")
                            rag_system.initialize_from_directory(pdf_dir, save_path)
                        
                        # Store in session state
                        st.session_state.rag_system = rag_system
                        st.session_state.documents_loaded = True
                        
                        st.success("🎉 System initialized successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error initializing system: {str(e)}")
        else:
            st.success("✅ System Ready")
            
            # Show document info
            if st.session_state.rag_system:
                total_docs = len(st.session_state.rag_system.documents_loaded)
                st.info(f"📚 Loaded: {total_docs} documents")
                
                if hasattr(st.session_state.rag_system.vector_store, 'index'):
                    total_chunks = st.session_state.rag_system.vector_store.index.ntotal
                    st.info(f"📊 Chunks: {total_chunks}")
            
            # Reset button
            if st.button("🔄 Reset System"):
                st.session_state.rag_system = None
                st.session_state.documents_loaded = False
                st.rerun()
        
        st.markdown("---")
        
        # Document list
        st.header("📁 Documents")
        if st.session_state.documents_loaded and st.session_state.rag_system:
            pdf_files = list(Path(pdf_dir).glob("*.pdf"))
            for pdf_file in pdf_files:
                st.text(f"📄 {pdf_file.name}")
        else:
            st.info("No documents loaded")
        
        st.markdown("---")
        
        # System info
        st.header("📋 System Info")
        st.text(f"📂 PDF Directory:")
        st.code(pdf_dir, language="text")
        st.text(f"💾 Index Path:")
        st.code(save_path, language="text")
    
    # Main area - Question and Answer
    if st.session_state.documents_loaded and st.session_state.rag_system:
        # Large, prominent question input
        st.markdown("## ❓ Ask Your Question")
        
        question = st.text_input(
            "",
            placeholder="What is the main topic discussed in the documents?",
            key="question_input",
            label_visibility="collapsed"
        )
        
        # Make the input box larger with custom CSS
        st.markdown("""
        <style>
        .stTextInput > div > div > input {
            font-size: 18px;
            padding: 15px;
            height: 60px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Answer section
        if question:
            with st.spinner("🔍 Searching documents..."):
                try:
                    result = st.session_state.rag_system.ask_question(question)
                    
                    # Display answer prominently
                    st.markdown("---")
                    st.markdown("## 💡 Answer")
                    
                    # Answer in a nice container
                    with st.container():
                        st.markdown(f"**Q:** {question}")
                        st.markdown("**A:**")
                        st.write(result['answer'])
                    
                    # Confidence and source info
                    col1, col2 = st.columns(2)
                    with col1:
                        confidence_color = "🟢" if result["confidence"] == "High" else "🟡"
                        st.markdown(f"**Confidence:** {confidence_color} {result['confidence']}")
                    
                    with col2:
                        st.markdown(f"**Sources:** {len(result['sources'])} document(s)")
                    
                    # Sources section
                    if result['sources']:
                        st.markdown("---")
                        st.markdown("## 📖 Sources")
                        
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"📄 Source {i}: {source['source_file']}", expanded=False):
                                st.text_area(
                                    "Content:",
                                    source["content"],
                                    height=150,
                                    key=f"source_content_{i}",
                                    disabled=True
                                )
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Please try rephrasing your question.")
        
        else:
            # Show example questions when no question is entered
            st.markdown("---")
            st.markdown("## 💡 Try These Example Questions:")
            st.info("👆 Copy any question below and paste it in the input field above")
            
            examples = [
                "What is the main topic of the documents?",
                "Can you summarize the key findings?",
                "What are the important policies mentioned?",
                "Who are the main people discussed?",
                "What dates or deadlines are mentioned?",
                "What are the key recommendations?"
            ]
            
            # Display examples in a nice format for easy copying
            for i, example in enumerate(examples, 1):
                with st.container():
                    col1, col2 = st.columns([1, 10])
                    with col1:
                        st.markdown(f"**{i}.**")
                    with col2:
                        st.code(example, language="text")
            
            st.markdown("---")
            st.info("💡 **Tip:** Select any question above and copy-paste it into the input field for quick testing!")
    
    else:
        # Initial state - big call to action
        st.markdown("---")
        st.markdown("## 🚀 Get Started")
        st.info("👈 Click 'Initialize System' in the sidebar to load your documents and start asking questions!")
        
        # Show what documents will be loaded
        if os.path.exists(pdf_dir):
            pdf_files = list(Path(pdf_dir).glob("*.pdf"))
            if pdf_files:
                st.markdown("### 📚 Documents Ready to Load:")
                for pdf_file in pdf_files:
                    st.markdown(f"- 📄 {pdf_file.name}")
            else:
                st.warning(f"No PDF files found in {pdf_dir}")
        else:
            st.error(f"PDF directory not found: {pdf_dir}")
import os
from dotenv import load_dotenv

load_dotenv()  # loads from .env if present
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Hardcoded Configuration for Testing
TESTING_CONFIG = {
    "API_KEY": OPENAI_API_KEY,
    "PDF_DIRECTORY": "/OwnAiProjects/pdfstore",
    "SAVE_PATH": "./pdf_index",
    "AUTO_SAVE": True
}

def create_test_interface():
    """Simple test interface with hardcoded paths"""
    
    print("📚 PDF Question Answering System (Test Mode)")
    print("=" * 50)
    
    # Use hardcoded configuration
    api_key = TESTING_CONFIG["API_KEY"]
    pdf_dir = TESTING_CONFIG["PDF_DIRECTORY"]
    save_path = TESTING_CONFIG["SAVE_PATH"]
    
    # Validate configuration
    if not api_key.startswith('sk-'):
        print("❌ Please update API_KEY in TESTING_CONFIG with your OpenAI API key")
        return
    
    if not os.path.exists(pdf_dir):
        print(f"❌ PDF directory '{pdf_dir}' does not exist")
        return
    
    # Check if PDFs exist
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in '{pdf_dir}'")
        return
    
    print(f"✅ Found {len(pdf_files)} PDF files in '{pdf_dir}'")
    for pdf_file in pdf_files:
        print(f"   📄 {pdf_file.name}")
    
    # Initialize system (use_streamlit=False for console output)
    print(f"\n🔄 Initializing system...")
    rag_system = PDFRAGSystem(api_key, use_streamlit=False)
    
    try:
        # Check if saved index exists
        if save_path and os.path.exists(save_path):
            print(f"📂 Found existing index at '{save_path}', loading...")
            rag_system.initialize_from_saved(save_path)
        else:
            print(f"🆕 Creating new index from PDFs...")
            rag_system.initialize_from_directory(pdf_dir, save_path if TESTING_CONFIG["AUTO_SAVE"] else None)
        
        # Interactive question-answering
        print("\n🎉 System ready! Type 'quit' to exit.")
        print("💡 Try asking: 'What is the main topic?' or 'Summarize the key points'")
        print("-" * 50)
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("   Please enter a question or type 'quit' to exit")
                continue
            
            try:
                print("🔍 Searching documents...")
                result = rag_system.ask_question(question)
                
                print(f"\n💡 **Answer:**")
                print(f"{result['answer']}")
                print(f"\n📊 **Confidence:** {result['confidence']}")
                print(f"📚 **Sources Used:** {len(result['sources'])} document(s)")
                
                # Show source files
                if result['sources']:
                    print(f"\n📖 **Source Files:**")
                    source_files = list(set(source['source_file'] for source in result['sources']))
                    for file in source_files:
                        print(f"   • {file}")
            
            except Exception as e:
                print(f"❌ Error answering question: {str(e)}")
                print("   Please try rephrasing your question")
    
    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        # Streamlit mode
        create_streamlit_app()
    else:
        # Test mode with hardcoded paths (default)
        create_test_interface()