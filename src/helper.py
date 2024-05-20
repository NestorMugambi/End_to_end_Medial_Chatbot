from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings



def load_pdf(data):
    loader = PyPDFDirectoryLoader(data
                    
                    )
    
    documents = loader.load()

    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

def cohere_embeddings():
    embeddings = CohereEmbeddings()

    return embeddings