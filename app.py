import streamlit as st
import cv2
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import sys

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.recognition.face_matcher import FaceMatcher

# --- Configuration ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "rgb_model.pth")
FACE_DETECTOR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224
MOTION_THRESHOLD = 6.0
STABLE_FRAME_LIMIT = 10
LIVE_DECAY_FRAMES = 5

preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- AI Classes ---
class LivenessDetector:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.prev_face = None
        self.stable_frames = 0
        self.live_counter = 0

    def _load_model(self, path):
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        checkpoint = torch.load(path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(DEVICE).eval()
        return model

    def check_liveness(self, face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img_t = preprocess(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(img_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            cls = int(probs.argmax())
            conf = float(probs[cls])
            label = "LIVE" if cls == 0 else "SPOOF"

        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        motion_score = 0
        if self.prev_face is not None:
            diff = cv2.absdiff(cv2.resize(face_gray, (100, 100)), cv2.resize(self.prev_face, (100, 100)))
            motion_score = np.mean(diff)
            if motion_score < MOTION_THRESHOLD:
                self.stable_frames += 1
            else:
                self.stable_frames = 0
        self.prev_face = face_gray.copy()

        if self.stable_frames > STABLE_FRAME_LIMIT and label == "LIVE":
            label = "SPOOF"
            conf = 0.65  

        if label == "LIVE":
            self.live_counter += 1
        else:
            self.live_counter = max(0, self.live_counter - 1)

        if self.live_counter < LIVE_DECAY_FRAMES and label == "LIVE":
            label = "UNSURE"

        return label, conf, motion_score

# --- Cache Models so Streamlit doesn't reload them every click ---
@st.cache_resource
def load_ai_engines():
    liveness_engine = LivenessDetector(MODEL_PATH)
    db_path = os.path.join(BASE_DIR, "data", "mock_kyc_db.json")
    kyc_matcher = FaceMatcher(db_path=db_path, threshold=0.40)
    face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_XML)
    return liveness_engine, kyc_matcher, face_cascade

# --- Scanner Logic ---
def run_scanner(liveness_engine, kyc_matcher, face_cascade):
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    kyc_throttle_counter = 0
    consecutive_verifications = 0
    verified_user_data = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-8)
        prev_time = curr_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

        current_kyc_text = "Scanning for faces..."
        kyc_box_color = (255, 255, 0)

        for (x, y, w, h) in faces:
            pad = int(0.15 * h)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_bgr = frame[y0:y1, x0:x1]

            label, conf, motion_score = liveness_engine.check_liveness(face_bgr)
            
            kyc_throttle_counter += 1
            if label == "LIVE" and kyc_throttle_counter % 5 == 0: 
                match_result = kyc_matcher.verify_identity(frame)
                if match_result["status"] == "success":
                    verified_user_data = match_result["data"]
                    current_kyc_text = f"Verified: {verified_user_data['name']}"
                    kyc_box_color = (0, 255, 0)
                    consecutive_verifications += 1
                else:
                    current_kyc_text = "Unknown User"
                    kyc_box_color = (0, 165, 255)
                    consecutive_verifications = 0
            elif label == "SPOOF":
                current_kyc_text = "SPOOF DETECTED"
                kyc_box_color = (0, 0, 255)
                consecutive_verifications = 0

            # UI Overlays
            color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
            if label == "UNSURE": color = (0, 255, 255)
            
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            liveness_text = f"{label} {conf*100:5.1f}% | motion={motion_score:.1f}"
            (text_w, text_h), _ = cv2.getTextSize(liveness_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x0, y0 - text_h - 8), (x0 + text_w + 6, y0), color, -1)
            cv2.putText(frame, liveness_text, (x0 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # Bottom Banner & Auto-Complete Check
        h_frame, w_frame, _ = frame.shape
        cv2.rectangle(frame, (0, h_frame - 50), (w_frame, h_frame), (0, 0, 0), -1)
        
        if consecutive_verifications > 0:
            progress = min(100, int((consecutive_verifications / 10) * 100))
            current_kyc_text += f" [Capturing Data: {progress}%]"

        cv2.putText(frame, current_kyc_text, (20, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, kyc_box_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("Automated e-KYC Scanner", frame)

        # Auto-break loop if verified successfully 10 times in a row!
        if consecutive_verifications >= 10:
            time.sleep(0.5) # Pause briefly so user sees 100%
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return verified_user_data

# --- STREAMLIT FRONTEND UI ---
def main():
    st.set_page_config(page_title="NextGen e-KYC Portal", layout="centered")

    # Initialize session state for multi-page flow
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None

    # Load Models
    liveness_engine, kyc_matcher, face_cascade = load_ai_engines()

    # --- PAGE 1: HOME ---
    if st.session_state.page == 'home':
        st.title("🛡️ Enterprise e-KYC Verification")
        st.write("Welcome to the AI-Powered Identity Verification Portal.")
        st.write("This system uses Multimodal Anti-Spoofing and Facial Recognition to securely extract KYC details without manual data entry.")
        
        st.info("Ensure you are in a well-lit area and looking directly at the camera.")
        
        if st.button("Start E-KYC Process", type="primary", use_container_width=True):
            st.session_state.page = 'scanning'
            st.rerun()

    # --- PAGE 2: SCANNING ---
    elif st.session_state.page == 'scanning':
        st.title("📷 Facial Verification In Progress")
        st.warning("A secure camera window will now open. Please look at the camera until the progress bar reaches 100%.")
        
        with st.spinner("Initializing Anti-Spoofing Engine & Camera..."):
            extracted_data = run_scanner(liveness_engine, kyc_matcher, face_cascade)
            
        if extracted_data:
            st.session_state.user_data = extracted_data
            st.success("Verification Successful! Processing KYC data...")
            time.sleep(1)
            st.session_state.page = 'form'
            st.rerun()
        else:
            st.error("Verification Aborted or Failed. Please try again.")
            if st.button("Return Home"):
                st.session_state.page = 'home'
                st.rerun()

    # --- PAGE 3: AUTO-FILLED FORM ---
    elif st.session_state.page == 'form':
        st.title("📝 Aadhaar KYC Details")
        st.success("Identity Verified. Data securely extracted from the mock Aadhaar database.")
        
        data = st.session_state.user_data
        
        # Display as a clean form
        with st.form("kyc_form"):
            st.text_input("Full Name", value=data.get("name", ""), disabled=True)
            st.text_input("Aadhaar Number (UID)", value=data.get("aadhaar_no", ""), disabled=True)
            st.text_input("Date of Birth", value=data.get("dob", ""), disabled=True)
            st.text_area("Registered Address", value=data.get("address", ""), disabled=True)
            
            st.write("---")
            submitted = st.form_submit_button("Confirm & Proceed", type="primary", use_container_width=True)
            
            if submitted:
                st.session_state.page = 'success'
                st.rerun()

    # --- PAGE 4: FINAL SUCCESS ---
    elif st.session_state.page == 'success':
        st.balloons()
        st.title("✅ Verification Complete!")
        st.success("User verified successfully. The e-KYC process is now finalized.")
        
        if st.button("Start New Verification Session"):
            st.session_state.page = 'home'
            st.session_state.user_data = None
            st.rerun()

if __name__ == "__main__":
    main()