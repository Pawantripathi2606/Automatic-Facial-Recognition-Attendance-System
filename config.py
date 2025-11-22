# config.py

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# SQLite DB path
DB_PATH = os.path.join(BASE_DIR, "attendance_system.db")

# Dataset folder (if needed later)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# App configs
APP_TITLE = "Automatic Facial Recognition Attendance System"

# DeepFace configs
# Model options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"  # or "retinaface" if you install extra deps

# Distance metric & threshold (for embeddings)
# We'll use cosine distance; lower is more similar
DISTANCE_METRIC = "cosine"
MATCH_THRESHOLD = 0.35  # tweak if needed (lower = stricter, higher = more lenient)
