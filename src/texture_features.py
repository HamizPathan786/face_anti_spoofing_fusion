# texture_features.py
import os
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm

def compute_lbp_hist(image_gray, P=8, R=1, bins=256):
    lbp = local_binary_pattern(image_gray, P, R, method="uniform")
    # histogram of LBP
    (hist, _) = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-8)
    return hist

def generate_lbp_dataset(input_root="../data/casia-fasd", output_root="../data/casia-fasd_lbp"):
    classes = ["train", "test"]
    labels = ["live", "spoof"]
    for split in classes:
        for lab in labels:
            in_dir = Path(input_root) / split / lab
            out_dir = Path(output_root) / split / lab
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in tqdm(sorted(list(in_dir.glob("*"))), desc=f"{split}/{lab}"):
                if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (224, 224))
                hist = compute_lbp_hist(img, P=8, R=1, bins=59)  # uniform patterns typically 59 bins
                np.save(out_dir / (p.stem + ".npy"), hist)

if __name__ == "__main__":
    generate_lbp_dataset()
    print("LBP feature generation done.")
