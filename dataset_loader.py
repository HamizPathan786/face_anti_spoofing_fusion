import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ✅ Correct relative paths
train_dir = os.path.join("..", "data", "casia-fasd", "train")
test_dir = os.path.join("..", "data", "casia-fasd", "test")

# ✅ Verify folder contents
print("Train folder:", train_dir)
print("Contains:", os.listdir(train_dir))

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ✅ Load dataset
dataset = datasets.ImageFolder(train_dir, transform=transform)
print(f"Found {len(dataset)} training images across {len(dataset.classes)} classes: {dataset.classes}")

if len(dataset) == 0:
    print("⚠️ No images found! Check your dataset folder paths.")
else:
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print("DataLoader created successfully!")


def default_transforms(train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8,1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

class CASIAFASDDataset(Dataset):
    """
    Expects folder structure:
    root_dir/
      live/
        *.jpg
      fake/
        *.jpg
    split is implemented by passing different root_dir paths (train/ test) or by random split in train script.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform is not None else default_transforms(train=False)
        self.samples = []
        classes = ["live", "spoof"]
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in glob.glob(os.path.join(cls_dir, ext)):
                    self.samples.append((p, label))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}. Check path/structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
    

if __name__ == "__main__":
    train_dir = os.path.join("..", "data", "casia-fasd", "train")
    print(f"Looking for training data in: {os.path.abspath(train_dir)}")

    dataset = CASIAFASDDataset(root_dir=train_dir, transform=default_transforms(train=True))
    print(f"✅ Found {len(dataset)} images total in {train_dir}")
    print(f"Classes: live = {sum(1 for _, l in dataset.samples if l==0)}, spoof = {sum(1 for _, l in dataset.samples if l==1)}")