from flask import Flask, render_template, jsonify, request
from src.helper import download_hf_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
# from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub

from dotenv import load_dotenv
from src.prompt import promptTemplate
import os


app = Flask(__name__)
load_dotenv()


embeddings = download_hf_embeddings()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "medical-chatbot"

try:
	# Attempt to connect to the existing Pinecone index
	docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
	print("Connected to existing Pinecone index.")

except Exception as e:
	print(f"Failed to connect to existing index: {e}")

PROMPT = PromptTemplate(template=promptTemplate, input=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}


model = "google/flan-t5-base"

# Initialize HuggingFaceHub with your API token
hf_hub = HuggingFaceHub(
    repo_id="google/flan-t5-base",
)

retriever = docsearch.as_retriever(search_kwargs={'k': 2});
qa_chain = RetrievalQA.from_chain_type(
	llm=hf_hub,
	chain_type="stuff",
	retriever=retriever,
	return_source_documents=True,
	chain_type_kwargs=chain_type_kwargs
)

@app.route('/')
def index():
	return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
	msg = request.form['msg']
	response = qa_chain({"query": msg})
	return str(response['result'])

if __name__ == '__main__':
	app.run(host='0.0.0.0', port='3000', debug= True)