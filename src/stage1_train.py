# stage1_train.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset_loader import CASIAFASDDataset, default_transforms
from model_rgb import Stage1Model
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score

def train_epoch(model, loader, criterion, opt, device):
    model.train()
    losses=[]; preds=[]; labs=[]
    for imgs, lbls in loader:
        imgs = imgs.to(device); lbls = lbls.to(device)
        opt.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds += torch.argmax(logits, dim=1).cpu().tolist()
        labs += lbls.cpu().tolist()
    return np.mean(losses), accuracy_score(labs, preds)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses=[]; preds=[]; labs=[]
    for imgs, lbls in loader:
        imgs = imgs.to(device); lbls = lbls.to(device)
        logits = model(imgs)
        loss = criterion(logits, lbls)
        losses.append(loss.item())
        preds += torch.argmax(logits, dim=1).cpu().tolist()
        labs += lbls.cpu().tolist()
    import numpy as np
    from sklearn.metrics import accuracy_score
    return np.mean(losses), accuracy_score(labs, preds)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CASIAFASDDataset(os.path.join(args.data_dir, "train"), transform=default_transforms(train=True))
    # split val if no val folder
    n = len(ds); n_val = int(0.15 * n); n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = Stage1Model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, opt, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss {tr_loss:.4f}, train_acc {tr_acc:.4f}, val_acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, os.path.join(args.save_dir, "stage1_best.pth"))
            print("Saved best stage1 model.")
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(args.save_dir, "stage1_final.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/casia-fasd")
    parser.add_argument("--save_dir", type=str, default="../saved_models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
