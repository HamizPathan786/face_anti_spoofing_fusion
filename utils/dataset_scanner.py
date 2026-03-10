import os
import cv2
import csv
import yaml
import random
from pathlib import Path
from collections import defaultdict

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return duration

def scan_dataset():
    config = load_config()
    root_path = Path(config['dataset']['root_path'])
    
    if not root_path.exists():
        print(f"Error: Dataset path does not exist at {root_path}")
        return
        
    print(f"Scanning dataset at: {root_path}")
    
    manifest_data = []
    stats = defaultdict(lambda: {'count': 0, 'duration': 0.0})
    
    # Recursively find all video files
    video_extensions = ['.avi', '.mp4', '.mov']
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(root_path.rglob(f"*{ext}"))
        
    for video_path in all_videos:
        rel_path = video_path.relative_to(root_path)
        path_parts = rel_path.parts
        
        # Determine label and attack type from path
        # CASIA standard structure: .../real/..., .../fake/warped_photo/...
        label_str = "real" if "real" in [p.lower() for p in path_parts] else "fake"
        label = 0 if label_str == "real" else 1
        
        attack_type = "none"
        if label == 1:
            for part in path_parts:
                if part.lower() in ["warped_photo", "cut_photo", "video_replay"]:
                    attack_type = part.lower()
                    break
            # If still fake but attack type not explicitly found, mark as unknown_spoof
            if attack_type == "none":
                attack_type = "unknown_spoof"
                
        duration = get_video_duration(video_path)
        stats[label_str]['count'] += 1
        stats[label_str]['duration'] += duration
        
        # Add to manifest list
        manifest_data.append({
            'video_path': str(video_path),
            'label': label,
            'attack_type': attack_type,
            'duration': duration
        })
        
    print("\n--- Dataset Summary ---")
    for key, val in stats.items():
        print(f"Class: {key.upper()} | Files: {val['count']} | Total Duration: {val['duration']:.2f} seconds")
        
    # Shuffle and split
    random.seed(42)
    random.shuffle(manifest_data)
    
    total = len(manifest_data)
    train_split = config['dataset']['train_split']
    val_split = config['dataset']['val_split']
    
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    for i, item in enumerate(manifest_data):
        if i < train_end:
            item['split'] = 'train'
        elif i < val_end:
            item['split'] = 'val'
        else:
            item['split'] = 'test'
            
    # Write to CSV
    manifest_path = Path("data/manifest.csv")
    manifest_path.parent.mkdir(exist_ok=True, parents=True)
    with open(manifest_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_path', 'label', 'attack_type', 'duration', 'split'])
        writer.writeheader()
        writer.writerows(manifest_data)
        
    print(f"\nManifest saved to {manifest_path} with {total} entries.")
    
if __name__ == "__main__":
    scan_dataset()
