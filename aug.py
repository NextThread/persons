import cv2
import numpy as np
import os
import random

INPUT_DIR = "dataset/train/person"
OUTPUT_DIR = "dataset/train/person"
TARGET_COUNT = 600

existing = [f for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(existing)} existing person images. Augmenting to {TARGET_COUNT}...")

count = len(existing)
while count < TARGET_COUNT:
    src_file = random.choice(existing)
    img = cv2.imread(os.path.join(INPUT_DIR, src_file))
    if img is None:
        continue

    # Random augmentations
    # 1. Flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # 2. Brightness
    factor = random.uniform(0.6, 1.4)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)

    # 3. Rotation
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # 4. Zoom
    scale = random.uniform(0.85, 1.15)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.resize(img, (128, 128))  # back to standard

    # 5. Gaussian noise
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    out_path = os.path.join(OUTPUT_DIR, f"aug_{count:04d}.jpg")
    cv2.imwrite(out_path, img)
    count += 1

print(f"✅ Done! Total person images: {count}")