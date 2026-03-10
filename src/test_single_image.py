import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# ==========================================================
# Configuration
# ==========================================================
model_path = "../models/rgb_model.pth"   # or saved_models/rgb_model.pth if you changed location
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same preprocessing used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================================
# Load Model
# ==========================================================
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)   # 2 classes: live / spoof
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ==========================================================
# Prediction Function
# ==========================================================
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
        prob = torch.softmax(outputs, dim=1)[0][pred].item()

    label = "LIVE" if pred.item() == 0 else "SPOOF"
    print(f"\n🧠 Prediction: {label}  (Confidence: {prob*100:.2f}%)")
    return label, prob

# ==========================================================
# Run Test
# ==========================================================
if __name__ == "__main__":
    test_image_path = "../data/test_unseen.jpg"   # <---- change this to your test image path
    label, confidence = predict_image(test_image_path)
