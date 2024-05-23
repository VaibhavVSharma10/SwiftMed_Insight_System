#   python 3.10 required 

# Importing HuggingFace embeddings for efficient and versatile NLP tasks
from langchain_community.embeddings import HuggingFaceEmbeddings

# Importing FAISS vector store for fast similarity search and clustering of dense vectors
from langchain_community.vectorstores import FAISS

# Importing document loaders for handling PDF documents and directories
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Importing text splitter for optimal chunking of text while maintaining context and coherence
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Create vector database

def create_vector_db():
    # Load PDF documents from the 'data/' directory using DirectoryLoader and PyPDFLoader
    loader = DirectoryLoader('data/',
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    # Load documents from the specified directory
    documents = loader.load()

    # Initialize text splitter to chunk documents into smaller pieces for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    # Split the loaded documents into smaller chunks
    texts = text_splitter.split_documents(documents)

    # Initialize HuggingFace embeddings model for converting text chunks into embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Create a FAISS vector store from the text chunks and their embeddings
    db = FAISS.from_documents(texts, embeddings)

    # Save the FAISS vector store locally
    db.save_local('vectorstore/db_faiss')

if __name__ == "__main__":
    # Execute the function to create and save the vector database
    create_vector_db()