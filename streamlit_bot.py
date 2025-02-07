import time
import os
import numpy as np
import faiss
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import streamlit as st


# Hosting local client (LM studio), work on openai API
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
embed = SentenceTransformer('bert-base-nli-mean-tokens')

# Loading CSV file
csv_file="clean_data.csv"
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
def chunk_data(docs, size=1000, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    doc = text_splitter.split_documents(docs)
    return doc
docs = chunk_data(document)

# Check for Vectorstore
index_path = "index.faiss"
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:

    # Creating the Vector Embeddings
    enbedding_list = []
    for i in range(len(docs)):
        resposnse = embed.encode(docs[i].page_content)
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

# Funtion for text response
def Chat(text, model, key, k):
    # User Input Embeddings
    start = time.time()
    text_embed = embed.encode([text])
    text_arr = np.array(text_embed)
    faiss.normalize_L2(text_arr)
    # Similarity Search
    D, I = index.search(text_arr, k=k)
    # Making the context
    string=[]
    for i in range(len(I[0])):
        string.append((docs[I[0][i]].page_content))
    context=" ".join(string)
    #funtion for calling chat model
    def call_openai_chat_model(api_key, model, user_input, context):
        OpenAI.api_key = api_key
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a college chatbot who have to provide information about the college provided as context"},
                {"role": "assistant", "content": context},
                {"role": "user", "content": user_input}
            ],
            max_tokens=200,
            stream=True
        )
        # print(response.choices[0].message.content)
        for event in response:
            print(event.choices[0].delta.content, end='', flush=True)
    # Chatbot Call
    call_openai_chat_model(api_key=key, model=model, user_input=text, context=context)
    chat_end = time.time()
    print(f"\nChat time: {chat_end-start} s\n")

st.title("AKGEC Chatbot")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    st.write(message)

# Input box and send button
user_input = st.text_input("You:", "")
if st.button("Send"):
    if user_input:
        st.session_state['messages'].append(f"You: {user_input}")
        bot_response = Chat(text=user_input, model="deepseek-r1-distill-qwen-1.5b", key="not-needed", k=2)
        st.session_state['messages'].append(f"Bot: {bot_response}")
        st.rerun()
