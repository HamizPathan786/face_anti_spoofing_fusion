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

class LivenessDetector:
    """Encapsulates the PyTorch model and the temporal motion logic."""
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.prev_face = None
        self.stable_frames = 0
        self.live_counter = 0

    def _load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}. Check MODEL_PATH.")
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
        # 1. PyTorch Prediction
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        img_t = preprocess(face_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(img_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            cls = int(probs.argmax())
            conf = float(probs[cls])
            label = "LIVE" if cls == 0 else "SPOOF"

        # 2. Motion Detection Logic
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        motion_score = 0
        if self.prev_face is not None:
            diff = cv2.absdiff(cv2.resize(face_gray, (100, 100)),
                               cv2.resize(self.prev_face, (100, 100)))
            motion_score = np.mean(diff)
            if motion_score < MOTION_THRESHOLD:
                self.stable_frames += 1
            else:
                self.stable_frames = 0
        self.prev_face = face_gray.copy()

        # 3. Temporal Smoothing & Decision
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

def main():
    print("Loading AI Models... Please wait.")
    
    # Initialize both engines
    liveness_engine = LivenessDetector(MODEL_PATH)
    db_path = os.path.join(BASE_DIR, "data", "mock_kyc_db.json")
    kyc_matcher = FaceMatcher(db_path=db_path, threshold=0.40)
    face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_XML)
    
    print("✅ Models loaded successfully!")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open webcam.")
        return

    print("Webcam started! Press 'q' to exit.")
    
    kyc_throttle_counter = 0
    current_kyc_text = "Scanning for faces..."
    kyc_box_color = (255, 255, 0)
    
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-8)
        prev_time = curr_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

        # Reset KYC text if nobody is in frame
        if len(faces) == 0:
            current_kyc_text = "No face detected"
            kyc_box_color = (255, 255, 0)

        for (x, y, w, h) in faces:
            # Add padding around the face
            pad = int(0.15 * h)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_bgr = frame[y0:y1, x0:x1]

            # --- STAGE 1: LIVENESS CHECK ---
            label, conf, motion_score = liveness_engine.check_liveness(face_bgr)
            
            # --- STAGE 2: KYC MATCHING (Only if LIVE) ---
            kyc_throttle_counter += 1
            if label == "LIVE" and kyc_throttle_counter % 5 == 0: 
                match_result = kyc_matcher.verify_identity(frame)
                if match_result["status"] == "success":
                    user = match_result["data"]
                    current_kyc_text = f"Verified: {user['name']} | UID: {user['aadhaar_no']}"
                    kyc_box_color = (0, 255, 0) # Green
                else:
                    current_kyc_text = "Unknown User in Database"
                    kyc_box_color = (0, 165, 255) # Orange
            elif label == "SPOOF":
                current_kyc_text = "KYC ABORTED - SPOOF DETECTED"
                kyc_box_color = (0, 0, 255) # Red

            # --- DRAW STAGE 1 UI (Bounding Box & Liveness) ---
            color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
            if label == "UNSURE": color = (0, 255, 255)
            
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            liveness_text = f"{label} {conf*100:5.1f}% | motion={motion_score:.1f}"
            (text_w, text_h), _ = cv2.getTextSize(liveness_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x0, y0 - text_h - 8), (x0 + text_w + 6, y0), color, -1)
            cv2.putText(frame, liveness_text, (x0 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # --- DRAW FPS (Top Left) ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # --- DRAW STAGE 2 UI (KYC Banner at Bottom) ---
        h_frame, w_frame, _ = frame.shape
        cv2.rectangle(frame, (0, h_frame - 50), (w_frame, h_frame), (0, 0, 0), -1)
        cv2.putText(frame, current_kyc_text, (20, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, kyc_box_color, 2)

        cv2.imshow("Production e-KYC & Anti-Spoofing Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()