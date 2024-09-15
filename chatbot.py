import streamlit as st
import google.generativeai as genai
import os
import json
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import creds

genai.configure(api_key=creds.GEMINI_API_KEY)

st.title("Know about our politicians? 🤔")

generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b-exp-0827",
    generation_config=generation_config
)

def load_data_from_folder(folder_path):
    data = []
    if not os.path.exists(folder_path):
        st.error(f"The folder '{folder_path}' does not exist.")
        return data
    
    files = os.listdir(folder_path)
    if not files:
        st.error(f"The folder '{folder_path}' is empty.")
        return data

    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                try:
                    json_data = json.load(file)
                    if "questions" in json_data:
                        for item in json_data["questions"]:
                            if isinstance(item, dict) and "question" in item:
                                data.append(item["question"])
                            else:
                                st.warning(f"No 'question' key found in one of the items in {filename}")
                    else:
                        st.warning(f"No 'questions' key found in file: {filename}")
                except json.JSONDecodeError:
                    st.error(f"Error decoding JSON from file: {filename}")

    return data

def prepare_input_text(data):
    return "\n".join(data)

def get_user_message_reply(user_message, data):
    try:
        election_data = prepare_input_text(data)
        
        prompt = (
            f"{election_data}\n"
            f"User Message: {user_message}\n"
            "Provide a comprehensive response based on the above election data and the user's message."
        )
        
        response = model.generate_content(
            [prompt],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return response.text
    except Exception as e:
        st.error(f"Error generating reply: {e}")
        return None
    
data_folder = "data1"
questions = load_data_from_folder(data_folder)
user_input = st.text_area("Do you have any doubts about our politicians? 🤔 If so, feel free to ask me! 😉 I’ll assist you with my knowledge.")

if st.button("Ask Help from AI"):
    if user_input and questions:
        with st.spinner("Wait AI will help you..."):
            response = get_user_message_reply(user_input, questions)
    
    if response:
        st.subheader("Response from Gemini AI")
        st.write(response)
