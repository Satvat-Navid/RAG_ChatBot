import time
import os
import numpy as np
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Hosting local client (LM studio), work on openai API
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv('GROQ_API_KEY'))
pc = Pinecone(api_key=os.getenv('PC_API_KEY'))
index = pc.Index('akgec-data')


# Loading CSV file
csv_file="data/clean_data.csv"
def read_doc(txt_file):
    loader = CSVLoader(file_path=txt_file,
        csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['Link', 'content']
    })
    document = loader.load()
    return document
document = read_doc(csv_file)
# Chunking the document
def chunk_data(docs, size=800, overlap=80):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    doc = text_splitter.split_documents(docs)
    return doc
chunks = chunk_data(document)

def response(query, k):
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )
    results = index.query(
        namespace="doc1",
        vector=query_embedding[0].values,
        top_k=k,
        include_values=False,
        include_metadata=True
    )
    return results

print("\nAKGEC CHATBOT: It will provide information about the college.\nType q to quit.")
while(True):
    # User Input Embeddings
    text = input("Question: ")
    start = time.time()
    if(text.lower() == 'q'):
        break

    top_k = 3
    # Similarity Search
    results = response(text, top_k)

    # Making the context
    string=[]
    for i in range(top_k):
        string.append((results["matches"][i]['metadata']['source_text']))
    context=" ".join(string)

    #funtion for calling chat model
    def call_openai_chat_model(api_key, model, user_input, context):
        OpenAI.api_key = api_key
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful chatbot, provide concise information from the context."},
                {"role": "user", "content": f"context : {context}"},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            stream=True
        )
        # print(response.choices[0].message.content)
        for event in response:
            print(event.choices[0].delta.content, end='', flush=True)

    # Chatbot Call
    call_openai_chat_model(api_key="no-needed", model="llama-3.3-70b-versatile", user_input=text, context=context)

    chat_end = time.time()
    print(f"\nChat time: {(chat_end-start):.2f} s\n")
