from src.helper import load_pdf,text_split,cohere_embeddings
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")


extracted_data = load_pdf("data")
text_chunks = text_split(extracted_data)
embeddings = cohere_embeddings()

index_name = "medicalchatbot"
docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)


