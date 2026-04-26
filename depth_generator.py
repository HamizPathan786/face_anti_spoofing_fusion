# depth_generator.py
# Run once to generate depth maps for your dataset.
def generate_midas_depth(input_root, output_root):
    import torch
    import glob
    import os
    from PIL import Image
    from tqdm import tqdm
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MiDaS model
    model_type = "MiDaS_small"  # lightweight & works on CPU too
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to(device)
    model.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform

    for split in ["train", "test"]:
        for cls in ["live", "spoof"]:
            input_dir = os.path.join(input_root, split, cls)
            output_dir = os.path.join(output_root, split, cls)
            os.makedirs(output_dir, exist_ok=True)

            print(f"Processing {input_dir} -> {output_dir}")
            for img_path in tqdm(glob.glob(os.path.join(input_dir, "*.jpg"))):
                try:
                    # 1️⃣ Load image and preprocess
                    img = Image.open(img_path).convert("RGB")
                    input_batch = transform(img).unsqueeze(0).to(device)

                    # 2️⃣ Forward pass
                    with torch.no_grad():
                        prediction = model(input_batch)
                        prediction = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=img.size[::-1],
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze()

                    # 3️⃣ Convert to depth map & save
                    depth_map = prediction.cpu().numpy()
                    depth_img = Image.fromarray((depth_map / depth_map.max() * 255).astype("uint8"))
                    save_path = os.path.join(output_dir, os.path.basename(img_path))
                    depth_img.save(save_path)

                except Exception as e:
                    print(f"⚠️ Error processing {img_path}: {e}")



if __name__ == "__main__":
    # WARNING: first-time run will download MiDaS weights and may take time.
    generate_midas_depth(input_root="../data/casia-fasd", output_root="../data/casia-fasd_depth")
    print("Depth generation completed.")
