import cv2
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


MODEL_PATH = "../models/rgb_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_DETECTOR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CONF_THRESHOLD_STAGE1 = 0.85
INPUT_SIZE = 224
MOTION_THRESHOLD = 6.0          # Controls spoof vs real movement sensitivity
STABLE_FRAME_LIMIT = 10         # Number of frames with no motion before flagging spoof
LIVE_DECAY_FRAMES = 5           # Stability window for live consistency check


preprocess = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_stage1_model(path):
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


def predict_face(model, face_pil):
    img_t = preprocess(face_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        cls = int(probs.argmax())
        conf = float(probs[cls])
        label = "LIVE" if cls == 0 else "SPOOF"
    return label, conf, cls

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Update MODEL_PATH variable.")

    print("Loading model...")
    model = load_stage1_model(MODEL_PATH)
    print("✅ Model loaded successfully.")

    face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_XML)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Cannot open webcam.")
        return

    print("Press 'q' to quit. Press 's' to save a snapshot (image + prediction).")
    saved_count = 0
    prev_time = time.time()
    prev_face = None
    stable_frames = 0
    live_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame capture failed. Exiting...")
                break

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-8)
            prev_time = curr_time

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

            for (x, y, w, h) in faces:
                pad = int(0.15 * h)
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(frame.shape[1], x + w + pad)
                y1 = min(frame.shape[0], y + h + pad)
                face_bgr = frame[y0:y1, x0:x1]
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)

                label, conf, _ = predict_face(model, face_pil)

               
                face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                motion_score = 0
                if prev_face is not None:
                    diff = cv2.absdiff(cv2.resize(face_gray, (100, 100)),
                                       cv2.resize(prev_face, (100, 100)))
                    motion_score = np.mean(diff)

                    if motion_score < MOTION_THRESHOLD:
                        stable_frames += 1
                    else:
                        stable_frames = 0
                prev_face = face_gray.copy()

                # --- Decision logic ---
                if stable_frames > STABLE_FRAME_LIMIT and label == "LIVE":
                    label = "SPOOF"
                    conf = 0.65  

                if label == "LIVE":
                    live_counter += 1
                else:
                    live_counter = max(0, live_counter - 1)

                
                if live_counter < LIVE_DECAY_FRAMES and label == "LIVE":
                    label = "UNSURE"

                color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
                if label == "UNSURE":
                    color = (0, 255, 255)

                
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                text = f"{label} {conf*100:5.1f}% | motion={motion_score:.1f}"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x0, y0 - text_h - 8), (x0 + text_w + 6, y0), color, -1)
                cv2.putText(frame, text, (x0 + 3, y0 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

           
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.imshow("Live Face Anti-Spoofing (Stage1 + Motion)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                out_path = f"../outputs/snap_{timestamp}.jpg"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, frame)
                print(f"📸 Saved snapshot: {out_path}")
                saved_count += 1

    except Exception as e:
        print(f"⚠️ Exception occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam released, windows closed safely.")


if __name__ == "__main__":
    main()