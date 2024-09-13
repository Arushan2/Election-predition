import streamlit as st
import pandas as pd
import os
import io
import requests

# Title of the app
st.title("Election Prediction App")

# Instructions for developers
st.subheader("Election Data from the 'data' Folder")
st.write("This app automatically loads all text files containing candidate data from the 'data' folder. The data will be cleaned and formatted for AI predictions.")

# Function to clean and process the file data
def process_file(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # Convert to StringIO for pandas
    string_io = io.StringIO(data)
    
    # Load into a dataframe, skipping bad lines
    df = pd.read_csv(string_io, sep=",", on_bad_lines='skip')
    
    # Clean the data
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # Strip whitespace from strings
    
    # Convert problematic columns to string to avoid serialization issues
    for column in df.columns:
        df[column] = df[column].astype(str)
    
    return df
# Load all text files from the 'data' folder
def load_files_from_data_folder():
    data_folder = './data'
    all_dataframes = []
    
    # Ensure the 'data' folder exists
    if not os.path.exists(data_folder):
        st.error("The 'data' folder is missing. Please create it and add the text files.")
        return None
    
    # Load all .txt files in the folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)
            df = process_file(file_path)
            all_dataframes.append(df)
    
    # Combine all dataframes into one (assuming all files have the same structure)
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df
    else:
        st.error("No valid text files found in the 'data' folder.")
        return None

# Function to call the Gemini API for prediction
def get_prediction(data):
    # Replace with your actual Gemini API endpoint and request structure
    api_url = "https://api.gemini.com/v1/predict"
    
    # Mock headers (add real headers if necessary)
    headers = {
        'Content-Type': 'application/json'
    }
    
    # Sending data to the API
    response = requests.post(api_url, json=data, headers=headers)
    
    # Print the status code and response for debugging
    st.write("Status Code:", response.status_code)
    st.write("Response Content:", response.text)
    
    # Handling the response from the API
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch predictions. Status Code: {response.status_code}")
        return None

# Load and display the data
st.write("Loading data from the 'data' folder...")
df = load_files_from_data_folder()

if df is not None:
    st.write("Loaded Candidate Data")
    st.write(df)
    
    # Convert the dataframe to a dictionary for API usage
    data_dict = df.to_dict(orient="records")

    # Button to trigger the prediction
    if st.button("Predict Election Outcome"):
        st.write("Fetching prediction...")
        prediction = get_prediction(data_dict)
        
        # Show the prediction if successful
        if prediction:
            st.subheader("Election Prediction Results")
            st.write(prediction)
        else:
            st.error("No prediction available. Check your input data or the API.")
else:
    st.write("Please ensure the 'data' folder contains valid text files.")