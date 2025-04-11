# rng_app.py (Streamlit Frontend)

import streamlit as st
import requests # Need requests library
from PIL import Image # Keep for displaying image if needed, or remove
import io
import os # Only needed if constructing API URL dynamically

# --- Configuration ---
# IMPORTANT: Replace with your ACTUAL Render API URL after deployment!
# It will look something like 'https://your-service-name.onrender.com/predict/'
# RENDER_API_URL = os.environ.get("RENDER_API_URL", "http://localhost:10000/predict/")
RENDER_API_URL = os.environ.get("RENDER_API_URL", "https://rng-strength-classifier.onrender.com")

if RENDER_API_URL == "YOUR_RENDER_API_URL_HERE/predict/":
     st.warning("API URL not set. Using placeholder.")

SEQ_LENGTH = 1024 # Keep for file upload check message

# --- Remove Model/Scaler Loading Functions ---
# (Delete load_my_model and load_my_scaler_params functions)

# ----------- Streamlit UI Definition -----------
st.title("RNG Strength Predictor (via API)")

st.write(f"Upload a binary file containing {SEQ_LENGTH} bytes of RNG data.")
uploaded_file = st.file_uploader("Choose a binary file", type=None) # Accept any file

predict_button = st.button("Predict Strength")

# ----------- Streamlit Logic (Updated to call API) -----------
if predict_button and uploaded_file is not None:
    if RENDER_API_URL == "YOUR_RENDER_API_URL_HERE/predict/":
         st.error("Please set the RENDER_API_URL in the script or environment variables.")
    else:
        sequence_bytes = uploaded_file.getvalue()

        if len(sequence_bytes) == SEQ_LENGTH:
            st.info(f"Sending {len(sequence_bytes)} bytes to prediction API...")
            try:
                # Send file to FastAPI endpoint
                files = {'file': (uploaded_file.name, sequence_bytes, uploaded_file.type or 'application/octet-stream')}
                response = requests.post(RENDER_API_URL, files=files, timeout=60) # Add timeout

                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

                # Process successful response
                api_result = response.json()
                prediction = api_result.get("prediction")
                probability = api_result.get("probability_flawed")

                st.subheader("Prediction Result:")
                st.write(f"The model predicts the RNG sequence is: **{prediction}**")
                if probability is not None:
                    st.progress(float(probability))
                    st.write(f"(Probability of being Flawed: {probability:.4f})")

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                try:
                     # Try to display error detail from API if available
                     error_detail = response.json().get('detail', 'No detail provided.')
                     st.error(f"API Error Detail: {error_detail}")
                except:
                     pass # Ignore if response cannot be parsed as JSON

        else:
            st.error(f"Uploaded file contains {len(sequence_bytes)} bytes. Please upload a file with exactly {SEQ_LENGTH} bytes.")

elif predict_button:
    st.warning("Please upload a file first.")

# Add instructions or examples
st.sidebar.header("Instructions")
st.sidebar.info(f"1. Upload a binary file containing exactly {SEQ_LENGTH} bytes.\n"
             f"2. Click the 'Predict Strength' button.\n"
             "3. The result ('Healthy' or 'Flawed') will be displayed.")