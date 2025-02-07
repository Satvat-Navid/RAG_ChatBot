import os
# import requests
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading data: {e}")
        return ""
    
def get_answer(prompt, context, api_key):
    API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-cased-distilled-squad"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": {
            "question": f"{prompt}",
            "context": f"{context}"},
        "parameters": {
            "top_k": 4
        },
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    final_result = result[0]["answer"]
    return final_result

input_text = load_data('context.txt')
api_key = os.getenv('HUGGINGFACE_API_KEY')

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
        bot_response = get_answer(user_input, input_text, api_key)
        st.session_state['messages'].append(f"Bot: {bot_response}")
        st.rerun()
