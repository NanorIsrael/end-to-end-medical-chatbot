from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import time
import os

from src.helper import download_hf_embeddings, get_or_create_pinecone_index, load_pdf, text_split

load_dotenv()


extracted_data = load_pdf('data/')
text_chunks = text_split(extracted_data)
embeddings = download_hf_embeddings()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "medical-chatbot"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768, 
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        ) 
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

docsearch = get_or_create_pinecone_index(index_name, text_chunks, embeddings)