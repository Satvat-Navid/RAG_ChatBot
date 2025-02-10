import time
import os
import numpy as np
import faiss
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# Hosting local client (LM studio), work on openai API
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
embed = SentenceTransformer('all-MiniLM-L6-v2')

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

# Check for Vectorstore
index_path = "data/index.faiss"
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    # Creating the Vector Embeddings
    enbedding_list = []
    for i in range(len(chunks)):
        resposnse = embed.encode(chunks[i].page_content)
        enbedding_list.append(resposnse)
    embeddings_df_arr = np.array(enbedding_list).astype('float32')

    # Creating Vectorstore
    vector_dimension = embeddings_df_arr.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    print(len(embeddings_df_arr))
    faiss.normalize_L2(embeddings_df_arr)
    index.add(embeddings_df_arr)
    faiss.write_index(index, index_path)

print("\nAKGEC CHATBOT: It will provide information about the college.\nType q to quit.")
while(True):
    # User Input Embeddings
    text = input("Question: ")
    start = time.time()
    if(text.lower() == 'q'):
        break
    text_embed = embed.encode([text])
    text_arr = np.array(text_embed).astype('float32')
    faiss.normalize_L2(text_arr)

    # Similarity Search
    D, I = index.search(text_arr, k=2)

    # Making the context
    string=[]
    for i in range(len(I[0])):
        string.append((chunks[I[0][i]].page_content))
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
    call_openai_chat_model(api_key="no-needed", model="deepseek-r1-distill-qwen-1.5b", user_input=text, context=context)

    chat_end = time.time()
    print(f"\nChat time: {chat_end-start} s\n")
