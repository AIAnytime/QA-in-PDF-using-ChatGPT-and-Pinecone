# Import necessary libraries and modules
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Pinecone and OpenAI API keys from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set OpenAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Document preprocessing function
@st.cache_data
def doc_preprocessing():
    # Load documents from directory and only select PDFs
    loader = DirectoryLoader(
        'data/',
         glob='**/*.pdf',
          show_progress=True)

    # Load the documents from the directory      
    docs = loader.load()
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
         chunk_overlap=0)

    # Split the loaded documents into smaller chunks    
    docs_split = text_splitter.split_documents(docs)
    
    return docs_split

# Embedding database function
@st.cache_resource
def embedding_db():
    # Use the openAI embedding model
    embeddings = OpenAIEmbeddings()
    
    # Initialize Pinecone with API key and environment
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV)
    
    # Preprocess documents
    docs_split = doc_preprocessing()
    
    # Create document database using Pinecone and preprocessed documents
    doc_db = Pinecone.from_documents(
        docs_split,
        embeddings,
        index_name='vercel'
        )
    
    return doc_db

# Create a Language Model
llm = ChatOpenAI(model='gpt-3.5-turbo')

# Create an embedding database
doc_db = embedding_db()

# Function for retrieving answer using a query
def retrieval_answer(query):
    # Create a RetrievalQA object using a language model and document retriever
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=doc_db.as_retriever())
    
    # Run the query
    result = qa.run(query)
    
    return result

# Main function for running the app
def main():
    # Set title of the app
    st.title("Wise Guydes")

    # Get user's input query
    text_input = st.text_input("Ask your query...")
    
    # Run the query when the "Ask Query" button is clicked
    if st.button("Ask Query"):
        # Check if the query is not empty
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)

# Run the main function when the script is run directly
if __name__ == "__main__":
    main()
