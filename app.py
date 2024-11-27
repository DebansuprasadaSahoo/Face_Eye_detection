import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Load Haar cascades
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Streamlit app title
st.title("Face and Eye Detector")

# Option to select image source
option = st.selectbox("Select image source", ("Upload an Image",))

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Convert the file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_classifier.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(100, 100)
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Region of interest (ROI) for eyes within the detected face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = image[y:y + h, x:x + w]

                # Detect eyes
                eyes = eye_classifier.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=10,
                    minSize=(30, 30)
                )
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert BGR to RGB for Streamlit display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the output
            st.image(image, caption='Processed Image', use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")

# Add custom CSS for background image
bg_image_url = "https://cdn.wallpapersafari.com/11/43/rPe2aI.jpg"  # Replace with your image URL
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_image_url}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
    }}
    </style>
    """, unsafe_allow_html=True)
