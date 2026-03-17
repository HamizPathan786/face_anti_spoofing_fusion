import json
import numpy as np
from deepface import DeepFace
import os

class FaceMatcher:
    def __init__(self, db_path="../data/mock_kyc_db.json", threshold=0.40):
        self.threshold = threshold
        self.known_encodings = []
        self.known_users = []
        
        if not os.path.exists(db_path):
            print(f"Error: Database not found at {db_path}")
            return
            
        with open(db_path, 'r') as f:
            self.database = json.load(f)
            
        for user_id, user_data in self.database.items():
            if "embedding" in user_data:
                self.known_encodings.append(np.array(user_data["embedding"]))
                self.known_users.append(user_data)
                
        print(f"FaceMatcher initialized: Loaded {len(self.known_users)} users from database.")

    def verify_identity(self, cv2_frame):
        try:
            # Extract embedding from the live webcam frame using Facenet
            # enforce_detection=False prevents crashes if you blink or move too fast
            live_objs = DeepFace.represent(img_path=cv2_frame, model_name="Facenet", enforce_detection=False)
            
            if not live_objs:
                return {"status": "error", "message": "No face detected"}
            
            live_embedding = np.array(live_objs[0]["embedding"])
            
            # Compare against everyone in the database using Cosine Distance
            distances = []
            for known_emb in self.known_encodings:
                cos_dist = 1 - np.dot(live_embedding, known_emb) / (np.linalg.norm(live_embedding) * np.linalg.norm(known_emb))
                distances.append(cos_dist)
            
            if not distances:
                return {"status": "failed", "message": "Database empty"}

            best_match_idx = np.argmin(distances)
            best_distance = distances[best_match_idx]

            # If the distance is lower than our strict threshold, it's a match!
            if best_distance <= self.threshold:
                user = self.known_users[best_match_idx]
                confidence = round((1 - best_distance) * 100, 2)
                return {"status": "success", "data": user, "confidence": confidence}
            else:
                return {"status": "failed", "message": "User not found"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}