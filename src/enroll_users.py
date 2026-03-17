from deepface import DeepFace
import os
import json
import numpy as np

# Bulletproof paths regardless of where you run the script from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_DIR = os.path.join(BASE_DIR, "data", "aadhaar_mock")
DB_PATH = os.path.join(BASE_DIR, "data", "mock_kyc_db.json")

# These keys now EXACTLY match your folder names from the screenshot!
MOCK_AADHAAR_INFO = {
    "Aryan": {
        "name": "Aryan", "aadhaar_no": "1111-2222-3333", "dob": "01-01-2001", "address": "Mumbai, MH"
    },
    "Dhruv": {
        "name": "Dhruv", "aadhaar_no": "4444-5555-6666", "dob": "02-02-2001", "address": "Thane, MH"
    },
    "Hamiz Pathan": {
        "name": "Hamiz Pathan", "aadhaar_no": "1234-5678-9012", "dob": "15-08-2001", "address": "Thane, MH"
    },
    "Harsh Mohite": {
        "name": "Harsh Mohite", "aadhaar_no": "7777-8888-9999", "dob": "03-03-2001", "address": "Mumbai, MH"
    },
    "Hussain": {
        "name": "Hussain", "aadhaar_no": "1010-2020-3030", "dob": "04-04-2001", "address": "Pune, MH"
    },
    "Sameet": {
        "name": "Sameet", "aadhaar_no": "4040-5050-6060", "dob": "05-05-2001", "address": "Mumbai, MH"
    },
    "Saniya": {
        "name": "Saniya", "aadhaar_no": "7070-8080-9090", "dob": "06-06-2001", "address": "Navi Mumbai, MH"
    },
    "Shashank": {
        "name": "Shashank", "aadhaar_no": "1212-3434-5656", "dob": "07-07-2001", "address": "Thane, MH"
    }
}

def enroll_users():
    print(f"Looking for images in: {MOCK_DIR}")
    print("Starting e-KYC Enrollment with DeepFace...")
    database = {}

    if not os.path.exists(MOCK_DIR):
        print(f"ERROR: Cannot find folder {MOCK_DIR}")
        return

    for person_name in os.listdir(MOCK_DIR):
        person_path = os.path.join(MOCK_DIR, person_name)
        
        # Skip if it's not a folder, or not in our dictionary
        if not os.path.isdir(person_path) or person_name not in MOCK_AADHAAR_INFO:
            continue

        print(f"\nProcessing images for: {person_name}")
        for image_name in os.listdir(person_path):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            img_path = os.path.join(person_path, image_name)
            try:
                # Extract face math!
                embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=True)[0]["embedding"]
                database[person_name] = MOCK_AADHAAR_INFO[person_name]
                database[person_name]["embedding"] = embedding
                print(f" -> Successfully enrolled {person_name}")
                break # We just need 1 good photo per person
            except Exception as e:
                print(f" -> ERROR processing {image_name}: {e}")

    # Save it explicitly to the data folder
    with open(DB_PATH, 'w') as f:
        json.dump(database, f, indent=4)
    print(f"\nEnrollment Complete! Saved exactly to {DB_PATH}")

if __name__ == "__main__":
    enroll_users()