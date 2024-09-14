import streamlit as st
import os
import json
import google.generativeai as genai
import pandas as pd
import creds
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure Gemini API with your key
genai.configure(api_key=creds.GEMINI_API_KEY)

# Title of the app
st.title("Election Prediction App")

# Configuration for the Generative Model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Load the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b-exp-0827",
    generation_config=generation_config
)

# Function to read JSON files from the 'data' folder
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
                    # Extract questions from the JSON data
                    if "questions" in json_data:
                        data.extend(json_data["questions"])
                    else:
                        st.warning(f"No 'questions' key found in file: {filename}")
                except json.JSONDecodeError:
                    st.error(f"Error decoding JSON from file: {filename}")

    return data

# Combine all data into a single string for prediction
def prepare_input_text(data):
    combined_text = ""
    for item in data:
        combined_text += f"Q: {item['text']}\nA: {item['answer']}\n\n"
    return combined_text

# Function to get predictions from Gemini AI model
def get_prediction(input_text):
    try:
        # Modify the prompt to request data in a format suitable for a bar chart
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
        
        # Convert the response to JSON
        response_data = json.loads(response.text)
        return response_data
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        return None

# Function to get a response from Gemini based on user's message and data
def get_user_message_reply(user_message, data):
    try:
        # Prepare the election data as context
        election_data = prepare_input_text(data)
        
        # Combine election data with user's message
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

# Load data from 'data' folder
data_folder = 'data'
data = load_data_from_folder(data_folder)

if data:
    input_text = prepare_input_text(data)
    
    # Button to trigger the prediction
    if st.button("Predict Election Outcome"):
        st.write("Fetching prediction from Gemini AI model...")
        prediction_data = get_prediction(input_text)
        
        # Display the prediction as a bar chart if successful
        if prediction_data:
            st.subheader("Election Prediction Results")
            candidates = prediction_data.get('candidates', [])
            percentages = prediction_data.get('percentages', [])
            
            if candidates and percentages:
                # Create a DataFrame for the bar chart
                df = pd.DataFrame({
                    'Candidates': candidates,
                    'Percentage': percentages
                })
                
                # Display the bar chart
                st.bar_chart(df.set_index('Candidates'))
            else:
                st.error("Invalid data format returned from Gemini API.")
        else:
            st.error("No prediction available. Check the data or the API.")
else:
    st.write("No data found in the 'data' folder.")

# Divider for UI separation
st.markdown("---")

# User message input
st.subheader("Send a Message to Gemini AI")
user_message = st.text_input("Enter your message")

# Button to send the user message
if st.button("Send Message"):
    if user_message:
        st.write("Sending your message to Gemini AI...")
        reply = get_user_message_reply(user_message, data)
        
        if reply:
            st.subheader("Gemini AI's Reply")
            st.write(reply)
        else:
            st.error("No reply received from Gemini AI.")
    else:
        st.error("Please enter a message before sending.")
