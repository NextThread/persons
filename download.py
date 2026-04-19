import os
import requests
import zipfile
import shutil
from pathlib import Path

# We'll use Open Images Dataset subset via a simple downloader
# and INRIA Person Dataset (free, research use)

def download_file(url, dest):
    print(f"⬇️  Downloading {dest}...")
    r = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Done: {dest}")

def setup_dirs():
    dirs = [
        "dataset/train/person",
        "dataset/train/non_person",
        "dataset/test/person",
        "dataset/test/non_person",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("📁 Directories created.")

setup_dirs()

print("""
=====================================================
  MANUAL DATASET DOWNLOAD INSTRUCTIONS
=====================================================

Your current dataset is too small (31 person, 25 non_person).
You need at least 500 per class. Here's where to get them FREE:

─────────────────────────────────────────────────
OPTION 1 (EASIEST): Download pre-cropped datasets
─────────────────────────────────────────────────

1. PERSON images (full body crops):
   → https://www.kaggle.com/datasets/tapakah68/pedestrian-detection
   → https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset
   Download and put images into: dataset/train/person/

2. NON-PERSON images (backgrounds, objects):
   → https://www.kaggle.com/datasets/prasunroy/natural-images
   (use categories: airplane, car, flower, mountain, building)
   Download and put images into: dataset/train/non_person/

─────────────────────────────────────────────────
OPTION 2: Use this script to auto-generate non_person crops
─────────────────────────────────────────────────
Run: python crop_negatives.py  (created below)

=====================================================
""")