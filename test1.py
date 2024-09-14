import streamlit as st
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import creds

# Configure Gemini API with your key
genai.configure(api_key=creds.GEMINI_API_KEY)

# Title of the app
st.title("Election Prediction App")

# Instructions for developers
st.subheader("Election Predictions using Gemini AI Model")
st.write("This app uses Google's Gemini AI model for making predictions based on your input. Please provide relevant data for accurate election outcome predictions.")

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

# User input section
st.subheader("Input Election Data")
input_data = ("Use the given data only for output. Add an sentence in every response \"Source:https://www.ihp.lk/research-updates/changes-support-leading-presidential-candidates-during-2024-lead-campaign-period\".",
  "Question Who is currently leading in support among the candidates?",
  "Answer President Ranil Wickremesinghe is currently leading in support among the candidates.",
  "Question Who is in the second place according to the August 2024 MRP model estimates?",
  "Answer Sajith Premadasa is in second place according to the August 2024 MRP model estimates.",
  "Question Who is trailing in third place based on the MRP projections?",
  "Answer Anura Dissanayake is trailing in third place based on the MRP projections.",
  "Question Who is the current frontrunner in the voting intent projections?",
  "Answer President Ranil Wickremesinghe is the current frontrunner in the voting intent projections.",
  "Question Which candidate has the highest support among older voters (60+ years)?",
  "Answer President Ranil Wickremesinghe has the highest support among older voters (60+ years).",
  "Question Which candidate has the lowest support among voters aged 18–29 years?",
  "Answer Anura Dissanayake has the lowest support among voters aged 18–29 years.",
  "Question Who has the highest increase in support among women from January/February 2024 to June/July 2024?",
  "Answer President Ranil Wickremesinghe has the highest increase in support among women.",
  "Question Which candidate has experienced the largest drop in support among Sinhala voters?",
  "Answer Anura Dissanayake has experienced the largest drop in support among Sinhala voters.",
  "Question Who is the top candidate among poor voters based on the latest MRP estimates?",
  "Answer Sajith Premadasa is the top candidate among poor voters based on the latest MRP estimates.",
  "Question Which candidate saw the smallest change in support among Muslims?",
  "Answer President Ranil Wickremesinghe saw the smallest change in support among Muslims.",
  "Question Who has gained the most support among better-off voters?",
  "Answer President Ranil Wickremesinghe has gained the most support among better-off voters.",
  "Question Which candidate is leading in support among voters who are unfavourable to the Aragalaya?",
  "Answer President Ranil Wickremesinghe is leading in support among voters who are unfavourable to the Aragalaya.",
  "Question Who experienced the biggest loss in support among voters with a favourable view of the Aragalaya?",
  "Answer Anura Dissanayake experienced the biggest loss in support among voters with a favourable view of the Aragalaya.",
  "Question Which candidate has the highest support increase among voters who voted for Gotabaya Rajapaksa in 2019?",
  "Answer President Ranil Wickremesinghe has the highest support increase among voters who voted for Gotabaya Rajapaksa in 2019.",
  "Question Who is the most popular candidate among Muslim voters according to the latest data?",
  "Answer Sajith Premadasa is the most popular candidate among Muslim voters according to the latest data.",
  "Question Who saw the greatest decline in support among estate/Indian Tamil voters?",
  "Answer Sajith Premadasa saw the greatest decline in support among estate/Indian Tamil voters.",
  "Question Which candidate has the highest support among voters aged 30–59 years?",
  "Answer President Ranil Wickremesinghe has the highest support among voters aged 30–59 years.",
  "Question Who is the leading candidate among voters who supported Sajith Premadasa in the 2019 Presidential Election?",
  "Answer Sajith Premadasa is the leading candidate among voters who supported him in the 2019 Presidential Election.",
  "Question Which candidate had the highest increase in support among voters aged 18–29 years?",
  "Answer Sajith Premadasa had the highest increase in support among voters aged 18–29 years.",
  "Question Who had the largest drop in support among poor voters?",
  "Answer Anura Dissanayake had the largest drop in support among poor voters.",
  "Question Who is in the second place among voters who were unfavourable to the Aragalaya?",
)

# Function to get predictions from Gemini AI model
def get_prediction(input_text):
    try:
        # Append the "Who will win?" prompt to the user's input
        prompt = f"{input_text}\nGive me the winning chance in order in list?"
        
        response = model.generate_content(
            [prompt],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return response.text
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        return None

# Button to trigger the prediction
if st.button("Predict Election Outcome"):
    st.write("Fetching prediction from Gemini AI model...")
    prediction = get_prediction(input_data)
    
    # Display the prediction if successful
    if prediction:
        st.subheader("Election Prediction Results")
        st.write(prediction)
    else:
        st.error("No prediction available. Check your input data or the API.")