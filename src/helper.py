from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore


def load_pdf(data):
	loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
	documents = loader.load()
	return documents


def text_split(extracted_data):
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
	return text_splitter.split_documents(extracted_data)

def download_hf_embeddings():
	model = "sentence-transformers/sentence-t5-base"
	embeddings = HuggingFaceBgeEmbeddings(model_name=model)
	return embeddings


# Function to connect to existing index or create a new one
def get_or_create_pinecone_index(index_name, text_chunks, embeddings):
    try:
        # Attempt to connect to the existing Pinecone index
        docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
        print("Connected to existing Pinecone index.")

        query = "What causes malaria"
        docs = docsearch.similarity_search(query)
        # print(docs[0].page_content)
        if len(docs) == 0:
             docsearch = PineconeVectorStore.from_texts(
            [t.page_content for t in text_chunks], embedding=embeddings, index_name=index_name
        )
    except Exception as e:
        print(f"Failed to connect to existing index: {e}")
        print("Creating a new Pinecone index...")
        # Create a new Pinecone index
        docsearch = PineconeVectorStore.from_texts(
            [t.page_content for t in text_chunks], embedding=embeddings, index_name=index_name
        )
        print("New Pinecone index created.")
    return docsearch