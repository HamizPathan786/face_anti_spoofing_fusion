# stage2_train.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dataset_loader import CASIAFASDDataset, default_transforms
from model_rgb import EmbeddingModel
from fusion_model import FusionModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class FusionDataset(Dataset):
    def __init__(self, rgb_paths, depth_paths, lbp_paths, labels, transform_rgb):
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.lbp_paths = lbp_paths
        self.labels = labels
        self.transform_rgb = transform_rgb

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import cv2
        rgb = cv2.imread(self.rgb_paths[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        from PIL import Image
        from torchvision import transforms as T
        img = Image.fromarray(rgb)
        img_t = self.transform_rgb(img)
        depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, (224,224)).astype("float32")/255.0
        depth_t = torch.tensor(depth).unsqueeze(0)
        lbp = np.load(self.lbp_paths[idx]).astype("float32")
        lbp_t = torch.tensor(lbp)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_t, depth_t, lbp_t, label

def build_file_lists(root_rgb="../data/casia-fasd", root_depth="../data/casia-fasd_depth", root_lbp="../data/casia-fasd_lbp", split="train"):
    rgb_list=[]; depth_list=[]; lbp_list=[]; labels=[]
    classes = [("live",0), ("spoof",1)]
    for cls, lab in classes:
        rgb_dir = os.path.join(root_rgb, split, cls)
        depth_dir = os.path.join(root_depth, split, cls)
        lbp_dir = os.path.join(root_lbp, split, cls)
        for fname in os.listdir(rgb_dir):
            if not fname.lower().endswith((".jpg",".jpeg",".png")): continue
            rgb_list.append(os.path.join(rgb_dir, fname))
            depth_list.append(os.path.join(depth_dir, os.path.splitext(fname)[0]+".png"))
            lbp_list.append(os.path.join(lbp_dir, os.path.splitext(fname)[0]+".npy"))
            labels.append(lab)
    return rgb_list, depth_list, lbp_list, labels

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_list, depth_list, lbp_list, labels = build_file_lists(args.data_rgb, args.data_depth, args.data_lbp, split="train")
    train_idx, val_idx = train_test_split(list(range(len(labels))), test_size=0.15, stratify=labels, random_state=42)
    def sub(list_, idxs): return [list_[i] for i in idxs]
    train_ds = FusionDataset(sub(rgb_list, train_idx), sub(depth_list, train_idx), sub(lbp_list, train_idx), sub(labels, train_idx), transform_rgb=default_transforms(train=True))
    val_ds = FusionDataset(sub(rgb_list, val_idx), sub(depth_list, val_idx), sub(lbp_list, val_idx), sub(labels, val_idx), transform_rgb=default_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # instantiate fusion model
    rgb_backbone = EmbeddingModel(pretrained=True).to(device)
    # get rgb backbone feature dim (forward a dummy)
    rgb_backbone.eval()
    import torch
    dummy = torch.randn(1,3,224,224).to(device)
    with torch.no_grad():
        feat = rgb_backbone(dummy)
    rgb_dim = feat.shape[1]
    fusion = FusionModel(rgb_backbone_out_dim=rgb_dim, lbp_dim=lbp_list and np.load(lbp_list[0]).shape[0] or 59, pretrained_rgb=False).to(device)
    # copy rgb backbone weights into fusion's rgb_emb_model
    fusion.rgb_emb_model.load_state_dict(rgb_backbone.state_dict(), strict=False)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(fusion.parameters(), lr=args.lr)

    best_acc=0.0
    for epoch in range(1, args.epochs+1):
        fusion.train()
        train_losses=[]
        for rgb, depth, lbp, lbl in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            rgb = rgb.to(device); depth = depth.to(device); lbp = lbp.to(device); lbl = lbl.to(device)
            opt.zero_grad()
            logits = fusion(rgb, depth, lbp)
            loss = criterion(logits, lbl)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        # validate
        fusion.eval()
        import numpy as np
        preds=[]; labs=[]
        with torch.no_grad():
            for rgb, depth, lbp, lbl in val_loader:
                rgb = rgb.to(device); depth = depth.to(device); lbp = lbp.to(device)
                logits = fusion(rgb, depth, lbp)
                preds += torch.argmax(logits, dim=1).cpu().tolist()
                labs += lbl.tolist()
        from sklearn.metrics import accuracy_score
        val_acc = accuracy_score(labs, preds)
        print(f"Epoch {epoch} train_loss {np.mean(train_losses):.4f} val_acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": fusion.state_dict(), "val_acc": best_acc}, os.path.join(args.save_dir, "fusion_best.pth"))
            print("Saved best fusion model.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_rgb", type=str, default="../data/casia-fasd")
    parser.add_argument("--data_depth", type=str, default="../data/casia-fasd_depth")
    parser.add_argument("--data_lbp", type=str, default="../data/casia-fasd_lbp")
    parser.add_argument("--save_dir", type=str, default="../saved_models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
