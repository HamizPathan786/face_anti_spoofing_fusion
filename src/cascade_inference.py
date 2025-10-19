# cascade_inference.py
import os
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms as T
from model_rgb import Stage1Model, EmbeddingModel
from fusion_model import FusionModel
from dataset_loader import default_transforms
import csv

def load_stage1(path, device):
    ckpt = torch.load(path, map_location=device)
    m = Stage1Model(pretrained=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()
    return m

def load_fusion(path, device, rgb_dim=512, lbp_dim=59):
    ckpt = torch.load(path, map_location=device)
    fusion = FusionModel(rgb_backbone_out_dim=rgb_dim, lbp_dim=lbp_dim, pretrained_rgb=False)
    fusion.load_state_dict(ckpt["model_state_dict"])
    fusion.to(device).eval()
    return fusion

def preprocess_img_pil(pil_img, train=False):
    t = default_transforms(train=False)
    return t(pil_img).unsqueeze(0)  # 1xCxHxW

def preprocess_depth(depth_path):
    d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    d = cv2.resize(d, (224,224)).astype("float32")/255.0
    return torch.tensor(d).unsqueeze(0).unsqueeze(0)  # 1x1xHxW

def preprocess_lbp(lbp_path):
    arr = np.load(lbp_path).astype("float32")
    return torch.tensor(arr).unsqueeze(0)  # 1xlbp_dim

def cascade_predict(image_path, stage1, fusion, device, depth_root="../data/casia-fasd_depth", lbp_root="../data/casia-fasd_lbp", threshold=0.85):
    pil = Image.open(image_path).convert("RGB")
    img_t = preprocess_img_pil(pil).to(device)
    with torch.no_grad():
        logits1 = stage1(img_t)
        probs1 = torch.softmax(logits1, dim=1).cpu().numpy()[0]
        pred1 = int(probs1.argmax())
        conf1 = float(probs1.max())
    # map classes 0->live,1->spoof
    if conf1 >= threshold:
        return {"stage":1, "pred":("live","spoof")[pred1], "conf":conf1}
    # else: run fusion
    # construct depth & lbp paths by filename
    fname = os.path.basename(image_path)
    stem = os.path.splitext(fname)[0]
    # attempt multiple locations
    depth_path = None
    for candidate in [os.path.join(depth_root,"train","live",stem+".png"), os.path.join(depth_root,"train","spoof",stem+".png"), os.path.join(depth_root,"test","live",stem+".png"), os.path.join(depth_root,"test","spoof",stem+".png")]:
        if os.path.exists(candidate):
            depth_path = candidate
            break
    lbp_path = None
    for candidate in [os.path.join(lbp_root,"train","live",stem+".npy"), os.path.join(lbp_root,"train","spoof",stem+".npy"), os.path.join(lbp_root,"test","live",stem+".npy"), os.path.join(lbp_root,"test","spoof",stem+".npy")]:
        if os.path.exists(candidate):
            lbp_path = candidate
            break
    # if missing, create quick versions (on the fly)
    if depth_path is None:
        # fallback: estimate depth using simple blur/edge heuristic -> create uniform depth (low info)
        depth_tensor = torch.zeros(1,1,224,224)
    else:
        depth_tensor = preprocess_depth(depth_path)
    if lbp_path is None:
        lbp_tensor = torch.zeros(1,59)
    else:
        lbp_tensor = preprocess_lbp(lbp_path)
    # fusion expects batch tensors
    rgb = img_t.to(device)
    depth_tensor = depth_tensor.to(device)
    lbp_tensor = lbp_tensor.to(device)
    with torch.no_grad():
        logits2 = fusion(rgb, depth_tensor, lbp_tensor)
        probs2 = torch.softmax(logits2, dim=1).cpu().numpy()[0]
        pred2 = int(probs2.argmax()); conf2 = float(probs2.max())
    return {"stage":2, "pred":("live","spoof")[pred2], "conf":conf2, "stage1_pred":("live","spoof")[pred1], "stage1_conf":conf1}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1 = load_stage1("../saved_models/stage1_best.pth", device)
    # discover rgb_dim by embedding a dummy
    import torch
    emb = EmbeddingModel(pretrained=False).to(device)
    with torch.no_grad():
        dummy = torch.randn(1,3,224,224).to(device)
        rgb_feat = emb(dummy)
    rgb_dim = rgb_feat.shape[1]
    fusion = load_fusion("../saved_models/fusion_best.pth", device, rgb_dim=rgb_dim, lbp_dim=59)
    img_path = "../data/casia-fasd/test/live/00001.jpg"  # example
    print(cascade_predict(img_path, stage1, fusion, device))
