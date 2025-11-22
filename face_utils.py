# face_utils.py

import cv2
import numpy as np
import json
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis

from db import (
    update_student_face_encoding,
    get_all_students_with_encodings,
    has_attendance_for_date,
    insert_attendance,
)

# -------------------------
# INIT ARC FACE MODEL
# -------------------------
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


# -------------------------
# HELPER: COSINE DISTANCE
# -------------------------
def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------
# ENCODE FACE (REGISTRATION)
# -------------------------
def encode_single_face_from_frame(frame):
    if frame is None:
        return None

    faces = app.get(frame)
    if len(faces) == 0:
        return None

    # pick the biggest face
    faces = sorted(faces, key=lambda f: f.bbox[2] - f.bbox[0], reverse=True)

    emb = faces[0].normed_embedding  # 512D embedding
    emb = emb.astype("float32")

    return emb


# -------------------------
# SAVE ENCODING TO DB
# -------------------------
def save_student_face_encoding(student_id, emb):
    enc_json = json.dumps(emb.tolist())
    update_student_face_encoding(student_id, enc_json)


# -------------------------
# LOAD KNOWN ENCODINGS
# -------------------------
def load_known_face_encodings():
    rows = get_all_students_with_encodings()

    ids = []
    names = []
    encs = []

    for r in rows:
        try:
            arr = np.array(json.loads(r["face_encoding"]), dtype="float32")
            ids.append(r["id"])
            names.append(r["name"])
            encs.append(arr)
        except:
            continue

    return ids, names, encs


# -------------------------
# RECOGNIZE MULTIPLE FACES
# -------------------------
MATCH_THRESHOLD = 0.35

def recognize_faces_in_frame(frame, known_encs, known_ids, known_names):
    results = []
    if frame is None:
        return results

    faces = app.get(frame)
    if len(faces) == 0:
        return results

    for f in faces:
        emb = f.normed_embedding.astype("float32")
        bbox = f.bbox.astype(int)
        top, left = bbox[1], bbox[0]
        bottom, right = bbox[3], bbox[2]

        # Match with known encodings
        best_dist = 10
        best_idx = -1

        for i, known_emb in enumerate(known_encs):
            d = cosine_distance(known_emb, emb)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_dist <= MATCH_THRESHOLD:
            sid = known_ids[best_idx]
            name = known_names[best_idx]
        else:
            sid = None
            name = "Unknown"

        results.append({
            "student_id": sid,
            "name": name,
            "location": (top, right, bottom, left),
            "distance": float(best_dist),
        })

    return results


# -------------------------
# DRAW FACE BOXES
# -------------------------
def draw_face_boxes(frame, results):
    for r in results:
        top, right, bottom, left = r["location"]
        name = r["name"]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    return frame


# -------------------------
# MARK ATTENDANCE
# -------------------------
def mark_attendance_from_results(results, user_id, already_today):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    logs = []
    new = []

    for r in results:
        sid = r["student_id"]
        name = r["name"]

        if sid is None:
            continue

        if sid in already_today:
            continue

        if has_attendance_for_date(sid, today):
            already_today.add(sid)
            logs.append(f"{name} already marked today.")
            continue

        insert_attendance(sid, today, time_now, "Present", user_id)
        logs.append(f"Marked {name} present at {time_now}")
        already_today.add(sid)
        new.append(sid)

    return new, logs
