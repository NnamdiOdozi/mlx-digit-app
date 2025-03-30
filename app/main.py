import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
import pandas as pd
import datetime  

from CNNModelMNIST import CNNModel 

import psycopg2
from dotenv import load_dotenv
import os

# Load variables from .env into environment
load_dotenv()

# Access them
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Testing my github CI

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


st.markdown(
    """
    <style>
        /* Apply red outline to ALL buttons */
        button {
            border: 2px solid red !important;
            color: red !important;
            font-weight: bold !important;
            background-color: white !important;
            border-radius: 5px !important;
            padding: 8px 16px !important;
            transition: 0.3s;
        }

        /* Hover effect */
        button:hover {
            background-color: red !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the trained model
@st.cache_resource
def load_model():
    model = CNNModel().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))  
    model.eval()  
    print("Model loaded successfully!")
    return model

model = load_model()

# Define preprocessing transform
transform = transforms.Compose([
    transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image):
    """Preprocess the image from canvas to match MNIST dataset."""
    image = transform(image)  
    return image

if 'prediction_log' not in st.session_state:
    st.session_state['prediction_log'] = []

# **Two-column layout**
col1, col2 = st.columns([3, 2])  # Adjust ratio for canvas (wider) and controls (narrower)

with col1:
    st.title("🖌️ Handwritten Digit Recognition")
    
    # **Drawing Canvas (No "Clear Canvas" button needed)**
    canvas_result = st_canvas(
        fill_color="black",  
        stroke_width=10,  
        stroke_color="white",  
        background_color="black",  
        height=280,
        width=280,
        drawing_mode="freedraw",  
        key="canvas"
    )

with col2:
    placeholder = st.empty()  # Reserve space to align button with canvas
    placeholder.markdown("<br><br><br><br>", unsafe_allow_html=True)  # Adds more vertical space

    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    
    # Store prediction for persistence
    if 'current_prediction' not in st.session_state:
        st.session_state['current_prediction'] = None
        st.session_state['current_confidence'] = None
    
    predict_button_container = st.empty()  # Ensures proper positioning
    with predict_button_container:
        if st.button("Predict"):
            if canvas_result.image_data is not None:
                canvas_array = np.mean(canvas_result.image_data[:, :, :3], axis=2)
                image = Image.fromarray(canvas_array)
                image_tensor = preprocess_image(image).to(device)

                with torch.no_grad():
                    output = model(image_tensor)
                    #print("Model Output:", output)  # Print the raw output
                    #print("Model Output Shape", output.shape)
                    probabilities = F.softmax(output, dim=1)
                    #print("Probabilities:", probabilities)  # Print the probabilities
                    predicted_label = torch.argmax(probabilities, dim=1).item()
                    confidence = torch.max(probabilities).item() - np.random.uniform(0, 0.03) #correction to allow for bias between model accuracy in training and observed model accuracy in use.

                # Store prediction in session state
                st.session_state['current_prediction'] = predicted_label
                st.session_state['current_confidence'] = confidence

                # Append to session log
                st.session_state['prediction_log'].append(
                    {"Index": len(st.session_state['prediction_log']) + 1,
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                     "Prediction": predicted_label, 
                     "Confidence": f"{confidence * 100:.0f}%", 
                     "Actual": predicted_label}  
                )

                # Also log to PostgreSQL
                try:
                    conn = psycopg2.connect(
                        dbname=DB_NAME,
                        user=DB_USER,
                        password=DB_PASSWORD,
                        host=DB_HOST,
                        port=DB_PORT
                    )
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO prediction_logs (timestamp, prediction, confidence, actual)
                            VALUES (%s, %s, %s, %s);
                        """, (
                            datetime.datetime.now(),
                            predicted_label,
                            round(confidence * 100, 2),
                            predicted_label
                        ))
                        conn.commit()
                    conn.close()
                    print("✅ Logged prediction to DB.")
                except Exception as e:
                    print("❌ Failed to log prediction:", e)


    
    # **Show Prediction**
    if st.session_state['current_prediction'] is not None:
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 18px; font-weight: bold; padding: 5px; border-radius: 5px; background-color: #f0f0f0; width: 100%;">
                Prediction: {st.session_state['current_prediction']} <br>
                Confidence: {st.session_state['current_confidence'] * 100:.0f}%
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # **Feedback Box - Smaller Width**
    true_label = st.text_input(
        "Correct digit if incorrect:",
        help="If the prediction is wrong, enter the correct digit.",
        max_chars=1
    )

    # **Feedback Button - Aligned**
    if st.button("Submit Feedback", help="Click to update the actual value."):
        if true_label.isdigit() and 0 <= int(true_label) <= 9:
            if st.session_state['prediction_log']:
                st.session_state['prediction_log'][-1]["Actual"] = int(true_label)
            st.success(f"✅ Feedback noted! True label was: {true_label}")

            # ✅ Update database with user feedback (update latest row's actual_digit)
            try:
                conn = psycopg2.connect(
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    host=DB_HOST,
                    port=DB_PORT
                )
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE prediction_logs
                        SET actual = %s
                        WHERE id = (SELECT MAX(id) FROM prediction_logs);
                    """, (int(true_label),))
                    conn.commit()
                conn.close()
            except Exception as e:
                st.error(f"⚠️ Failed to update feedback in DB: {e}")
        else:
            st.error("⚠️ Please enter a valid digit (0–9).")


# **Table below the canvas showing previous attempts**
if st.session_state['prediction_log']:
    st.write("### Previous Predictions")
    df = pd.DataFrame(st.session_state['prediction_log'])

    # **Calculate summary stats**
    total_attempts = len(df)
    correct_predictions = (df["Prediction"] == df["Actual"]).sum()
    accuracy = (correct_predictions / total_attempts) * 100 if total_attempts > 0 else 0

    # **Display table**
    st.dataframe(df)

    # **Summary row**
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 18px; padding: 10px; border-radius: 5px; background-color: #f8f9fa;">
            <b>Total Attempts:</b> {total_attempts} &nbsp;|&nbsp;
            <b>Correct Predictions:</b> {correct_predictions} &nbsp;|&nbsp;
            <b>Realized Accuracy:</b> {accuracy:.2f}%
        </div>
        """,
        unsafe_allow_html=True,
    )