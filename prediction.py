import streamlit as st
import os
import json
import google.generativeai as genai
import pandas as pd
import creds
from google.generativeai.types import HarmCategory, HarmBlockThreshold

genai.configure(api_key=creds.GEMINI_API_KEY)

st.title("Election Predictions")

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
                        data.extend(json_data["questions"])
                    else:
                        st.warning(f"No 'questions' key found in file: {filename}")
                except json.JSONDecodeError:
                    st.error(f"Error decoding JSON from file: {filename}")

    return data
def prepare_input_text(data):
    combined_text = ""
    for item in data:
        combined_text += f"Q: {item['text']}\nA: {item['answer']}\n\n"
    return combined_text
def get_prediction(input_text):
    try:
        prompt = (
            f"{input_text}\n"
            "Act like an election prediction agent and give the approximate percentage for each candidate based on my data. "
            "Return the results in a JSON format as {'candidates': ['Candidate1', 'Candidate2'], 'percentages': [50, 50]}."
        )
        
        response = model.generate_content(
            [prompt],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        response_data = json.loads(response.text)
        return response_data
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        return None

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

data_folder = 'data'
data = load_data_from_folder(data_folder)

if data:
    input_text = prepare_input_text(data)
    if st.button("Predict Election Outcome"):
        st.write("Fetching prediction from Gemini AI model...")
        prediction_data = get_prediction(input_text)
        if prediction_data:
            st.subheader("My Election Prediction Results based on my knowledge")
            candidates = prediction_data.get('candidates', [])
            percentages = prediction_data.get('percentages', [])
            
            if candidates and percentages:
                df = pd.DataFrame({
                    'Candidates': candidates,
                    'Percentage': percentages
                })
                st.bar_chart(df.set_index('Candidates'))
            else:
                st.error("Invalid data format returned from Gemini API.")
        else:
            st.error("No prediction available. Check the data or the API.")
else:
    st.write("No data found in the 'data' folder.")
st.markdown("---")
st.subheader("Ask me about my prediction")
user_message = st.text_input("Eg:- Who can get most yougers vote?")
if st.button("Send Message"):
    if user_message:
        st.write("Sending your message to Gemini AI...")
        reply = get_user_message_reply(user_message, data)
        
        if reply:
            st.subheader("Here is my prediction")
            st.write(reply)
        else:
            st.error("No reply received from Gemini AI.")
    else:
        st.error("Please enter a message before sending.")
