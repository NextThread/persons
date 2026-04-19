import cv2
import numpy as np
import os
import random

# Put any background images (no people) in this folder
BACKGROUND_DIR = "backgrounds"
OUTPUT_DIR = "dataset/train/non_person"
TARGET_COUNT = 600
CROP_SIZE = (128, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BACKGROUND_DIR, exist_ok=True)

bg_images = [f for f in os.listdir(BACKGROUND_DIR)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not bg_images:
    print("⚠️  No background images found in 'backgrounds/' folder.")
    print("    Add some images without people (nature, rooms, streets etc.)")
    exit()

count = 0
while count < TARGET_COUNT:
    img_file = random.choice(bg_images)
    img = cv2.imread(os.path.join(BACKGROUND_DIR, img_file))
    if img is None:
        continue

    H, W = img.shape[:2]
    if H < 128 or W < 128:
        continue

    # Random crop
    x = random.randint(0, W - 128)
    y = random.randint(0, H - 128)
    crop = img[y:y+128, x:x+128]

    out_path = os.path.join(OUTPUT_DIR, f"neg_{count:04d}.jpg")
    cv2.imwrite(out_path, crop)
    count += 1

print(f"✅ Generated {count} non_person crops in {OUTPUT_DIR}/")