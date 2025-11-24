import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

from config import APP_TITLE
from db import (
    init_db,
    create_student,
    get_students,
    get_student_by_username,
    update_student,
    delete_student,
    update_attendance_status,
    delete_attendance_record,
    delete_all_attendance,
    clear_all_face_encodings,
)
from auth import (
    init_session_state,
    login_user,
    register_new_user,
    logout_user,
    require_login,
    require_role,
)
from face_utils import (
    encode_single_face_from_frame,
    save_student_face_encoding,
    load_known_face_encodings,
    recognize_faces_in_frame,
    draw_face_boxes,
    mark_attendance_from_results,
)
from attendance_utils import (
    mark_manual_attendance,
    attendance_to_dataframe,
    calculate_attendance_summary,
    get_attendance_dataframe,
    get_student_monthly_graph,
    get_class_wise_summary,
    get_daily_attendance_summary,
)

# OPTIONAL extra modules (uncomment if you created these files)
from timetable import timetable_page, init_timetable_table
from heatmap_utils import heatmap_page
from student_portal import student_portal_page
from multirole import get_allowed_pages_for_role
  # advanced multi-role menus

# ============================================================
# GLOBALS / HELPERS
# ============================================================

MOTION_LIVENESS_THRESHOLD = 7.0  # simple motion-based liveness


def compute_motion_score(prev_gray, curr_gray, location):
    """Very simple anti-spoof: mean abs diff in face ROI."""
    if prev_gray is None or curr_gray is None:
        return 0.0

    top, right, bottom, left = location
    h, w = curr_gray.shape[:2]

    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)

    if bottom <= top or right <= left:
        return 0.0

    roi_curr = curr_gray[top:bottom, left:right]
    roi_prev = prev_gray[top:bottom, left:right]

    if roi_curr.size == 0 or roi_prev.size == 0:
        return 0.0

    diff = cv2.absdiff(roi_curr, roi_prev)
    return float(np.mean(diff))


if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Login / Signup"


# ============================================================
# THEME + SMALL UI HELPERS
# ============================================================

def inject_theme():
    """Modern SaaS-style light blue theme."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #39A6FF 0%, #1488CC 40%, #0C75D2 100%);
            color: #0f172a;
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        }
        .block-container {
            padding-top: 0.9rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1200px !important;
        }
        [data-testid="stSidebar"] {
            background: rgba(15,23,42,0.08) !important;
            border-right: 1px solid rgba(15,23,42,0.08);
        }
        [data-testid="stSidebar"] > div {
            padding-top: 0.8rem;
        }
        .white-card {
            background: white;
            border-radius: 22px;
            padding: 22px 24px;
            box-shadow: 0 18px 50px rgba(15,23,42,0.35);
        }
        .soft-card {
            background: white;
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 14px 35px rgba(15,23,42,0.25);
        }
        .kpi-wrapper {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }
        .camera-frame {
            border-radius: 24px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.35);
            box-shadow: 0 24px 65px rgba(15,23,42,0.55);
        }
        .stButton > button {
            background: #0077ff;
            color: white;
            border-radius: 999px;
            border: none;
            padding: 0.45rem 1.4rem;
            font-weight: 600;
            box-shadow: 0 10px 25px rgba(15,23,42,0.35);
        }
        .stButton > button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_divider():
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(148,163,184,0.7);margin:1.4rem 0;'>",
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN APP ENTRY
# ============================================================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_theme()
    init_db()
    # timetable tables if you use timetable module
    try:
        init_timetable_table()
    except Exception:
        # ignore if timetable.py not present or function missing
        pass

    init_session_state()

    user = st.session_state.get("user")

    # Top hero header (like marketing page)
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start;
                    margin-bottom:0.9rem;">
            <div style="max-width:620px;">
                <p style="margin:0;color:#e0f2fe;font-size:13px;letter-spacing:0.15em;
                          text-transform:uppercase;font-weight:600;">
                    Smart camera-based attendance
                </p>
                <h1 style="color:white;margin:4px 0 4px;font-size:32px;font-weight:800;">
                    {APP_TITLE}
                </h1>
                <p style="color:#e0f2fe;margin:0;font-size:14px;">
                    Fool-proof, camera-based attendance tracking – because
                    <em>'I was here'</em> doesn’t work anymore.
                </p>
            </div>
            <div style="text-align:right;color:#e0f2fe;font-size:11px;">
                <div>Automatic Facial Recognition</div>
                <div>Multi-Face · Anti-Spoof · Analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # NAVIGATION
    if not st.session_state.get("is_authenticated"):
        menu = ["Login / Signup"]
    else:
        role = user["role"]

        # If advanced roles_advanced.py exists, use dynamic pages
        try:
            role_pages = get_allowed_pages_for_role(role)
            # Ensure Dashboard is first if included
            if "Dashboard" in role_pages:
                role_pages.remove("Dashboard")
                menu = ["Dashboard"] + role_pages
            else:
                menu = ["Dashboard"] + role_pages
        except Exception:
            # fallback to simple static menu
            if role == "admin":
                menu = [
                    "Dashboard",
                    "Mark Attendance (Camera)",
                    "Register Student & Capture Face",
                    "Train / Load Encodings",
                    "Manual Attendance",
                    "Attendance Reports",
                    "Student Management",
                    "Attendance Management",
                    "Admin Analytics",
                    "Live Admin Monitor",
                    "Insights & Alerts",
                    "Database Control Center",
                    "Timetable & Period-wise Attendance",
                    "Classroom Heatmap",
                ]
            elif role == "teacher":
                menu = [
                    "Dashboard",
                    "Mark Attendance (Camera)",
                    "Register Student & Capture Face",
                    "Train / Load Encodings",
                    "Manual Attendance",
                    "Attendance Reports",
                    "Student Management",
                    "Timetable & Period-wise Attendance",
                    "Classroom Heatmap",
                ]
            else:  # student
                menu = [
                    "Dashboard",
                    "Attendance Reports",
                    "Student Self-Service Portal",
                ]

    with st.sidebar:
        st.markdown("### Menu")
        choice = st.selectbox(
            "Navigation",
            menu,
            index=menu.index(st.session_state.get("current_page", menu[0])),
        )
        st.session_state["current_page"] = choice

        if st.session_state.get("is_authenticated"):
            st.markdown("---")
            st.markdown(
                f"**{user['full_name']}**  \nRole: `{user['role']}`",
                unsafe_allow_html=False,
            )
            if st.button("Logout"):
                logout_user()
                st.session_state["current_page"] = "Login / Signup"
                st.rerun()

    # ROUTING
    if choice == "Login / Signup":
        login_signup_page()
    elif choice == "Dashboard":
        if not st.session_state.get("is_authenticated"):
            st.warning("Please login first.")
            login_signup_page()
        else:
            role = st.session_state["user"]["role"]
            if role == "admin" or role == "principal":
                admin_dashboard()
            elif role == "teacher" or role == "supervisor":
                teacher_dashboard()
            else:
                student_dashboard()
    elif choice == "Mark Attendance (Camera)":
        mark_attendance_camera_page()
    elif choice == "Register Student & Capture Face":
        register_student_page()
    elif choice == "Train / Load Encodings":
        train_encodings_page()
    elif choice == "Manual Attendance":
        manual_attendance_page()
    elif choice == "Attendance Reports":
        attendance_reports_page()
    elif choice == "Student Management":
        student_management_page()
    elif choice == "Attendance Management":
        attendance_management_page()
    elif choice == "Admin Analytics":
        admin_analytics_dashboard()
    elif choice == "Live Admin Monitor":
        live_admin_monitor_page()
    elif choice == "Insights & Alerts":
        insights_alerts_page()
    elif choice == "Database Control Center":
        database_control_center_page()
    elif choice == "Timetable & Period-wise Attendance":
        timetable_page()
    elif choice == "Classroom Heatmap":
        heatmap_page()
    elif choice == "Student Self-Service Portal":
        student_portal_page()


# ============================================================
# LOGIN / SIGNUP
# ============================================================

def login_signup_page():
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("🔐 Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    # Login tab
    with tab1:
        st.markdown("#### Welcome back")
        uname = st.text_input("Username (Roll No for students)")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login_user(uname, pwd)
            if user:
                st.session_state["user"] = user
                st.session_state["is_authenticated"] = True
                st.session_state["current_page"] = "Dashboard"
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Invalid username or password")

    # Signup tab
    with tab2:
        st.markdown("#### Create a new account")
        full_name = st.text_input("Full Name")
        new_username = st.text_input("New Username (Student Roll No = Username)")
        new_password = st.text_input("New Password", type="password")
        # NOTE: for advanced roles, you can extend this list
        role = st.selectbox("Role", ["teacher", "admin", "student"])

        if st.button("Create Account"):
            success = register_new_user(new_username, new_password, full_name, role)
            if success:
                st.success("Account created. Please login.")
            else:
                st.error("Username already exists")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DASHBOARDS
# ============================================================

def admin_dashboard():
    # allow principal also to use admin dashboard
    require_role(["admin", "principal"])
    user = st.session_state["user"]

    # Get stats
    df = get_attendance_dataframe()
    students = get_students(None, None)
    total_students = len(students)

    total_records = len(df) if not df.empty else 0
    present_count = (df["status"] == "Present").sum() if not df.empty else 0
    absent_count = (df["status"] == "Absent").sum() if not df.empty else 0
    total_logs = present_count + absent_count
    avg_attendance = f"{(present_count / total_logs * 100):.1f}%" if total_logs > 0 else "N/A"

    today = datetime.now().strftime("%Y-%m-%d")
    if not df.empty:
        today_df = df[df["date"] == today]
        today_present = (today_df["status"] == "Present").sum()
        today_absent = total_students - today_present if total_students else 0
    else:
        today_present, today_absent = 0, 0

    left, right = st.columns([3, 2])

    # Left hero card
    with left:
        st.markdown(
            """
            <div class="white-card">
                <p style="color:#64748b;font-size:12px;font-weight:600;
                          letter-spacing:0.18em;text-transform:uppercase;margin:0 0 4px;">
                    Attendance that actually works
                </p>
                <h2 style="font-size:26px;line-height:1.1;margin:0 0 8px;
                           font-weight:800;color:#020617;">
                    Fool-proof smart attendance tracking
                </h2>
                <p style="font-size:14px;color:#1f2933;margin:0 0 10px;">
                    Because <em>'I was here'</em> doesn’t work anymore! Use AI-powered facial
                    recognition to simplify staff, student and classroom attendance.
                </p>
                <p style="font-size:13px;color:#4b5563;margin:0;">
                    Built-in biometric intelligence, instant records and analytics ready for payroll
                    or academic reports – all in one place.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Right snapshot + KPIs all fully visible
    with right:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <h3 style="margin:0 0 4px;font-size:18px;font-weight:800;color:#020617;">
                Quick system snapshot
            </h3>
            <p style="margin:0 0 12px;font-size:12px;color:#4b5563;">
                See how your attendance engine is performing today.
            </p>
            """,
            unsafe_allow_html=True,
        )

        k1, k2 = st.columns(2)
        k1.metric("Total Students", total_students)
        k2.metric("All-time Records", total_records)

        k3, k4 = st.columns(2)
        k3.metric("Present (All-time)", present_count)
        k4.metric("Avg Attendance", avg_attendance)

        k5, k6 = st.columns(2)
        k5.metric("Present Today", today_present)
        k6.metric("Approx. Absent Today", today_absent)

        st.markdown("</div>", unsafe_allow_html=True)

    section_divider()

    if not df.empty:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("#### Latest Attendance Events")
        st.dataframe(df.head(12))
        st.markdown("</div>", unsafe_allow_html=True)


def teacher_dashboard():
    # allow 'supervisor' to share teacher dashboard
    require_role(["teacher", "supervisor"])
    user = st.session_state["user"]
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader(f"📚 Teacher Dashboard – {user['full_name']}")
    st.info("Use sidebar options to manage students, train encodings and mark attendance.")
    st.markdown("</div>", unsafe_allow_html=True)


def student_dashboard():
    require_role(["student"])
    user = st.session_state["user"]

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader(f"🎓 Student Dashboard – {user['full_name']}")

    student = get_student_by_username(user["username"])
    if student is None:
        st.error("No student profile found. Your username must be your roll number.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    sid = student["id"]

    st.markdown("#### Profile")
    st.json(
        {
            "Roll No": student["student_id"],
            "Name": student["name"],
            "Class": student["class"],
            "Section": student["section"],
            "Email": student["email"],
        }
    )

    df = attendance_to_dataframe(student_id=sid)
    if df.empty:
        st.info("No attendance records yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    section_divider()

    st.markdown("#### Monthly Attendance")
    monthly = get_student_monthly_graph(df, sid)
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=monthly, x="month", y="Present Days", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("No monthly data available.")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ADMIN ANALYTICS
# ============================================================

def admin_analytics_dashboard():
    require_role(["admin", "principal"])
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("📈 Admin Analytics")

    df = get_attendance_dataframe()
    if df.empty:
        st.info("No attendance data.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df["date"] = pd.to_datetime(df["date"])
    present_count = (df["status"] == "Present").sum()
    total = len(df)
    avg_attendance = f"{present_count / total * 100:.1f}%" if total > 0 else "N/A"

    c1, c2 = st.columns(2)
    c1.metric("Total Records", total)
    c2.metric("Average Attendance", avg_attendance)

    section_divider()
    st.markdown("#### Daily Trend")
    daily = get_daily_attendance_summary(df)
    if not daily.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.lineplot(data=daily, x="date", y="Present Count", marker="o", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    section_divider()
    st.markdown("#### Class-wise Summary")
    class_summary = get_class_wise_summary(df)
    pres = class_summary[class_summary["status"] == "Present"]
    if not pres.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.barplot(data=pres, x="class", y="count", hue="section", ax=ax)
        st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# STUDENT REGISTRATION + ENCODINGS
# ============================================================

def register_student_page():
    require_role(["admin", "teacher"])
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("🧑‍🎓 Register Student & Capture Face")

    with st.form("regform"):
        sid = st.text_input("Roll No")
        name = st.text_input("Name")
        cls = st.text_input("Class")
        sec = st.text_input("Section")
        email = st.text_input("Email")
        submit = st.form_submit_button("Register")

    if submit:
        success, sid_db = create_student(sid, name, cls, sec, email)
        if success:
            st.success("Student added.")
            st.session_state["last_student_id"] = sid_db
        else:
            st.error("Roll No already exists.")

    section_divider()
    st.markdown("### 📷 Capture Face")

    sid = st.session_state.get("last_student_id")
    if not sid:
        st.info("Register student first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    img = st.camera_input("Capture")
    if img:
        data = np.frombuffer(img.read(), np.uint8)
        frame = cv2.imdecode(data, 1)
        enc = encode_single_face_from_frame(frame)
        if enc is None:
            st.error("No face detected.")
        else:
            save_student_face_encoding(sid, enc)
            st.success("Face encoding saved.")
    st.markdown("</div>", unsafe_allow_html=True)


def train_encodings_page():
    require_role(["admin", "teacher"])
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("🧠 Train / Load Face Encodings")

    if st.button("Load Encodings"):
        ids, names, encs = load_known_face_encodings()
        st.session_state["known_ids"] = ids
        st.session_state["known_names"] = names
        st.session_state["known_encodings"] = encs
        st.success(f"Loaded {len(ids)} encodings.")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# AUTO-REGISTER UNKNOWN FACE UI
# ============================================================

def auto_register_unknown_face_ui():
    face_img = st.session_state.get("pending_unknown_face", None)
    if face_img is None:
        return

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown("### 🧩 Unknown Face Detected – Auto Register")

    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Detected face (Unknown)", width=260)

    with st.form("auto_register_form"):
        roll_no = st.text_input("Roll No")
        name = st.text_input("Name")
        cls = st.text_input("Class")
        sec = st.text_input("Section")
        email = st.text_input("Email")
        submit = st.form_submit_button("Save & Register")

    if submit:
        if not roll_no or not name:
            st.error("Roll No and Name are required.")
        else:
            success, sid_db = create_student(roll_no, name, cls, sec, email)
            if not success or sid_db is None:
                st.error("Could not create student. Maybe roll no already exists.")
            else:
                emb = encode_single_face_from_frame(face_img)
                if emb is None:
                    st.error("Could not extract face embedding from captured face.")
                else:
                    save_student_face_encoding(sid_db, emb)
                    st.success(f"Student {name} registered and face saved ✅")
                    st.session_state["pending_unknown_face"] = None
                    ids, names, encs = load_known_face_encodings()
                    st.session_state["known_ids"] = ids
                    st.session_state["known_names"] = names
                    st.session_state["known_encodings"] = encs
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CAMERA ATTENDANCE – AUTO CAPTURE (NO FREEZE)
# ============================================================

def mark_attendance_camera_page():
    """
    Auto-capture based attendance:
    - NO continuous video loop (so no freezing)
    - On button click: capture short burst of frames
    - Use last frame to recognise faces
    - Mark attendance, show logs, done
    """
    require_role(["admin", "teacher", "supervisor"])
    st.subheader("📸 Auto-Capture Attendance (Single Snapshot)")

    known_ids = st.session_state.get("known_ids", [])
    known_encs = st.session_state.get("known_encodings", [])
    known_names = st.session_state.get("known_names", [])

    if not known_encs:
        st.warning("Please load face encodings first from 'Train / Load Encodings'.")
        return

    if "marked_today" not in st.session_state:
        st.session_state.marked_today = set()
    if "pending_unknown_face" not in st.session_state:
        st.session_state.pending_unknown_face = None

    # If previous capture had unknown face, ask to register first
    if st.session_state.pending_unknown_face is not None:
        auto_register_unknown_face_ui()
        st.info("After registering this face, click **Capture & Mark Attendance** again.")
        return

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size:14px;color:#4b5563;margin-bottom:10px;">
        Click the button below. The system will automatically capture a photo,
        recognise all faces in it and mark attendance – no live video needed.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Capture & Mark Attendance"):
        # ---- 1) Open camera and grab a short burst ----
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Could not open camera. Please check your webcam.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        frames = []
        for _ in range(5):
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frames.append(frame.copy())
            time.sleep(0.15)

        cap.release()

        if not frames:
            st.error("Failed to capture image from camera.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        frame = frames[-1]
        first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) if len(frames) > 1 else None
        last_gray = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)

        # ---- 2) Recognise faces ----
        results = recognize_faces_in_frame(frame, known_encs, known_ids, known_names)

        for r in results:
            if first_gray is not None:
                score = compute_motion_score(first_gray, last_gray, r["location"])
                r["motion_score"] = score
                r["is_live"] = score >= MOTION_LIVENESS_THRESHOLD
            else:
                r["motion_score"] = 0.0
                r["is_live"] = True

        live_known_faces = [r for r in results if r["student_id"] is not None and r["is_live"]]
        spoof_suspects = [r for r in results if r["student_id"] is not None and not r["is_live"]]
        live_unknown_faces = [r for r in results if r["student_id"] is None and r["is_live"]]

        logs = []

        # ---- 3) Mark attendance for live known faces ----
        if live_known_faces:
            _, new_logs = mark_attendance_from_results(
                live_known_faces,
                st.session_state["user"]["id"],
                st.session_state.marked_today,
            )
            if new_logs:
                logs.extend(new_logs)
        else:
            logs.append("ℹ️ No known (registered) faces detected in this snapshot.")

        # ---- 4) Spoof suspects (very low motion) ----
        for r in spoof_suspects:
            logs.append(
                f"⚠️ Spoof suspected for {r['name']} "
                f"(very low motion: {r['motion_score']:.1f}). Attendance not marked."
            )

        # ---- 5) Auto-register unknown live face (largest face) ----
        if live_unknown_faces:
            unk = sorted(
                live_unknown_faces,
                key=lambda x: (x["location"][2] - x["location"][0])
                * (x["location"][1] - x["location"][3]),
                reverse=True,
            )[0]
            top, right, bottom, left = unk["location"]
            h, w = frame.shape[:2]
            top = max(0, top)
            left = max(0, left)
            bottom = min(h, bottom)
            right = min(w, right)

            if bottom > top and right > left:
                face_crop = frame[top:bottom, left:right].copy()
                st.session_state.pending_unknown_face = face_crop
                logs.append(
                    "🆕 Unknown face detected. "
                    "Camera stopped – please fill the form below to register this person."
                )

        # ---- 6) Show captured image + logs ----
        frame_render = draw_face_boxes(frame, results)
        st.markdown('<div class="camera-frame">', unsafe_allow_html=True)
        st.image(cv2.cvtColor(frame_render, cv2.COLOR_BGR2RGB), channels="RGB")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Result log")
        for line in logs:
            if line.startswith("✅") or "present" in line.lower():
                st.success(line)
            elif line.startswith("⚠️") or "spoof" in line.lower():
                st.warning(line)
            else:
                st.info(line)

        if st.session_state.pending_unknown_face is not None:
            section_divider()
            auto_register_unknown_face_ui()
    else:
        st.info("Press **Capture & Mark Attendance** to take a snapshot and mark attendance.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MANUAL ATTENDANCE + REPORTS
# ============================================================

def manual_attendance_page():
    require_role(["admin", "teacher"])
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("✍️ Manual Attendance")

    cls = st.text_input("Class")
    sec = st.text_input("Section")
    date = st.date_input("Date").strftime("%Y-%m-%d")

    students = get_students(cls or None, sec or None)

    with st.form("manual_form"):
        st.markdown("##### Quick Actions")
        quick = st.radio(
            "Apply to students with 'Not Set' status:",
            ["None", "Mark all Present", "Mark all Absent"],
            horizontal=True,
        )

        status_map = {}
        st.markdown("##### Students")
        for s in students:
            status = st.selectbox(
                f"{s['student_id']} - {s['name']}",
                ["Not Set", "Present", "Absent"],
                key=f"s_{s['id']}",
            )
            status_map[s["id"]] = status

        save = st.form_submit_button("Save")

    if save:
        count = 0
        for sid, stv in status_map.items():
            if stv == "Not Set":
                if quick == "Mark all Present":
                    stv = "Present"
                elif quick == "Mark all Absent":
                    stv = "Absent"
            if stv != "Not Set":
                mark_manual_attendance(
                    sid,
                    stv,
                    date,
                    st.session_state["user"]["id"],
                )
                count += 1

        st.success(f"Saved {count} attendance records.")
    st.markdown("</div>", unsafe_allow_html=True)


def attendance_reports_page():
    require_login()
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("📜 Attendance Reports")

    roll = st.text_input("Roll No (optional)")
    cls = st.text_input("Class (optional)")
    sec = st.text_input("Section (optional)")

    sd = st.date_input("Start Date", None)
    ed = st.date_input("End Date", None)

    start = sd.strftime("%Y-%m-%d") if sd else None
    end = ed.strftime("%Y-%m-%d") if ed else None

    student_id = None
    if roll:
        st_data = get_student_by_username(roll)
        if st_data:
            student_id = st_data["id"]

    if st.button("Generate Report"):
        df = attendance_to_dataframe(student_id, cls or None, sec or None, start, end)
        if df.empty:
            st.info("No data found.")
        else:
            st.dataframe(df)
            summ = calculate_attendance_summary(df)
            st.write("Summary")
            st.dataframe(summ)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "attendance.csv")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# STUDENT & ATTENDANCE MANAGEMENT (CRUD)
# ============================================================

def student_management_page():
    require_role(["admin", "teacher"])
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("🧑‍🏫 Student Management (Edit / Delete)")

    cls = st.text_input("Filter by Class (optional)")
    sec = st.text_input("Filter by Section (optional)")

    students = get_students(cls or None, sec or None)
    if not students:
        st.info("No students found.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for s in students:
        with st.expander(f"{s['student_id']} - {s['name']}"):
            col1, col2 = st.columns(2)
            new_roll = col1.text_input("Roll No", value=s["student_id"], key=f"roll_{s['id']}")
            new_name = col2.text_input("Name", value=s["name"], key=f"name_{s['id']}")

            col3, col4, col5 = st.columns(3)
            new_cls = col3.text_input("Class", value=s["class"] or "", key=f"class_{s['id']}")
            new_sec = col4.text_input("Section", value=s["section"] or "", key=f"sec_{s['id']}")
            new_email = col5.text_input("Email", value=s["email"] or "", key=f"email_{s['id']}")

            colu1, colu2 = st.columns(2)
            if colu1.button("Update", key=f"upd_{s['id']}"):
                update_student(s["id"], new_roll, new_name, new_cls, new_sec, new_email)
                st.success("Student updated.")
                st.rerun()

            if colu2.button("Delete", key=f"del_{s['id']}"):

                delete_att = st.checkbox(
                    "Also delete this student's attendance records",
                    key=f"chk_{s['id']}",
                    value=False,
                )
                delete_student(s["id"], delete_attendance=delete_att)
                st.warning("Student deleted.")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def attendance_management_page():
    require_role(["admin", "principal", "hr"])
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("🧾 Attendance Management (Edit / Delete)")

    cls = st.text_input("Filter Class (optional)")
    sec = st.text_input("Filter Section (optional)")

    sd = st.date_input("From Date", None)
    ed = st.date_input("To Date", None)
    start = sd.strftime("%Y-%m-%d") if sd else None
    end = ed.strftime("%Y-%m-%d") if ed else None

    df = attendance_to_dataframe(None, cls or None, sec or None, start, end)
    if df.empty:
        st.info("No attendance data for given filters.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("#### Editable Attendance Records")
    for _, row in df.head(50).iterrows():
        with st.expander(
            f"#{row['id']} | {row['roll_no']} - {row['name']} | {row['date']} {row['time']}"
        ):
            current_status = row["status"]
            new_status = st.selectbox(
                "Status",
                ["Present", "Absent"],
                index=0 if current_status == "Present" else 1,
                key=f"att_status_{row['id']}",
            )

            c1, c2 = st.columns(2)
            if c1.button("Update", key=f"att_upd_{row['id']}"):
                update_attendance_status(row["id"], new_status)
                st.success("Attendance updated.")
                st.rerun()

            if c2.button("Delete", key=f"att_del_{row['id']}"):
                delete_attendance_record(row["id"])
                st.warning("Attendance record deleted.")
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# LIVE MONITOR + INSIGHTS + DB CONTROL
# ============================================================

def live_admin_monitor_page():
    require_role(["admin", "principal", "supervisor"])
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("🛰 Live Admin Monitor")

    df = get_attendance_dataframe()
    if df.empty:
        st.info("No attendance data yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    today = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df["date"] == today]

    total_classes = today_df["class"].nunique()
    total_present = (today_df["status"] == "Present").sum()
    total_absent = (today_df["status"] == "Absent").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Classes Today", total_classes)
    c2.metric("Present Today", total_present)
    c3.metric("Absent Records Today", total_absent)

    section_divider()
    st.markdown("#### Latest 20 Attendance Events (Today)")
    st.dataframe(today_df.head(20))
    st.markdown("</div>", unsafe_allow_html=True)


def insights_alerts_page():
    require_role(["admin", "principal"])
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("🧠 Insights & Alerts")

    df = get_attendance_dataframe()
    if df.empty:
        st.info("No attendance data yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    student_summary = (
        df.groupby(["student_id", "name", "roll_no", "class", "section", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    if "Present" not in student_summary.columns:
        student_summary["Present"] = 0
    if "Absent" not in student_summary.columns:
        student_summary["Absent"] = 0

    student_summary["Total"] = student_summary["Present"] + student_summary["Absent"]
    student_summary["Attendance %"] = np.where(
        student_summary["Total"] > 0,
        student_summary["Present"] / student_summary["Total"] * 100,
        0,
    )

    alert_threshold = st.slider("Alert threshold (%)", 50, 100, 75, 1)
    low_alert = student_summary[
        student_summary["Attendance %"] < alert_threshold
    ].sort_values("Attendance %")

    st.markdown(f"### 🚨 Students below {alert_threshold}% attendance")
    if low_alert.empty:
        st.success("No students below the threshold. 🎉")
    else:
        st.dataframe(
            low_alert[
                ["roll_no", "name", "class", "section", "Present", "Absent", "Attendance %"]
            ]
        )

    section_divider()
    st.markdown("#### Overall Attendance Distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(student_summary["Attendance %"], bins=10, kde=True, ax=ax)
    ax.set_xlabel("Attendance %")
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)


def database_control_center_page():
    require_role(["admin", "principal"])
    st.subheader("🗄 Database Control Center (Admin Only)")

    tab_students, tab_attendance, tab_danger = st.tabs(
        ["Students (CRUD)", "Attendance (CRUD)", "Danger Zone"]
    )

    with tab_students:
        student_management_page()

    with tab_attendance:
        attendance_management_page()

    with tab_danger:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown("#### 🔥 Danger Zone (Bulk Actions)")
        st.warning("These actions **cannot be undone**. Use carefully.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("❌ Delete ALL Attendance Records"):
                delete_all_attendance()
                st.success("All attendance records deleted successfully.")
        with c2:
            if st.button("🧹 Clear ALL Face Encodings"):
                clear_all_face_encodings()
                st.session_state["known_ids"] = []
                st.session_state["known_names"] = []
                st.session_state["known_encodings"] = []
                st.success("All face encodings cleared. Please re-register faces.")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
