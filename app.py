import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, WebRtcStreamer

# ... (rest of your code)

class FaceEyeDetector(VideoTransformerBase):
  """
  This class processes webcam frames to detect faces and eyes.
  """
  def __init__(self):
    # Load Haar cascades (replace with your path if not using default location)
    self.face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    self.eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

  def transform(self, frame):
    # ... (rest of your transform function)
    # Load Haar cascades (replace with your path if not using default location)
    self.face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    self.eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

  def transform(self, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = self.face_classifier.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=3, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        roi_gray = frame[y:y + h, x:x + w]
        eyes = self.eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
            cv2.putText(frame, "Eye", (x + ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Create a Streamlit app and WebRTC streamer
webrtc_streamer = WebRtcStreamer(key="face_eye_detector", video_transformer=FaceEyeDetector())
st.title("Face and Eye Detector")
st.write("This app detects faces and eyes in your webcam feed.")
st.write("Note: This functionality requires access to your webcam.")

if webrtc_streamer.media_stream:
    frame = webrtc_streamer.read_frame()
    # Display the processed frame
    st.image(frame, caption="Processed Webcam Feed", use_column_width=True)

# Add custom CSS for background image (unchanged)
# ... (rest of the background image code)
