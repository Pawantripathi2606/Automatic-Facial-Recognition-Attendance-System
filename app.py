# app.py

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


# ------------------------------ INITIAL STATE ------------------------------

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Login / Signup"


# ------------------------------ UI UTILITIES ------------------------------

def kpi_card(label, value, help_text=""):
    st.metric(label, value, help_text)

def section_divider():
    st.markdown("---")


# ------------------------------ MAIN APP ------------------------------

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
    )

    init_db()
    init_session_state()

    st.markdown(
        f"""
        <div style="background:linear-gradient(90deg,#1e3c72,#2a5298);
        padding:12px;border-radius:12px;margin-bottom:15px;color:white;">
            <h2 style="margin:0;">{APP_TITLE}</h2>
            <p style="margin:0;font-size:13px;opacity:0.9;">
                Facial Recognition Attendance System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user = st.session_state.get("user")

    # ---------------- MENU CONTROL ----------------
    if not st.session_state.get("is_authenticated"):
        menu = ["Login / Signup"]
    else:
        role = user["role"]
        if role == "admin":
            menu = [
                "Dashboard",
                "Register Student & Capture Face",
                "Train / Load Encodings",
                "Mark Attendance (Camera)",
                "Manual Attendance",
                "Attendance Reports",
                "Admin Analytics",
            ]
        elif role == "teacher":
            menu = [
                "Dashboard",
                "Register Student & Capture Face",
                "Train / Load Encodings",
                "Mark Attendance (Camera)",
                "Manual Attendance",
                "Attendance Reports",
            ]
        else:  # student
            menu = [
                "Dashboard",
                "Attendance Reports",
            ]

    with st.sidebar:
        st.markdown("### Navigation")

        choice = st.selectbox(
            "Menu",
            menu,
            index=menu.index(st.session_state.get("current_page", menu[0])),
        )
        st.session_state["current_page"] = choice

        if st.session_state.get("is_authenticated"):
            st.markdown("---")
            st.markdown(
                f"**User:** {user['full_name']}<br>"
                f"Role: {user['role']}",
                unsafe_allow_html=True
            )
            if st.button("Logout"):
                logout_user()
                st.session_state["current_page"] = "Login / Signup"
                st.rerun()

    # ---------------- ROUTER ----------------
    if choice == "Login / Signup":
        login_signup_page()
    elif choice == "Dashboard":
        if not st.session_state.get("is_authenticated"):
            st.warning("Please login first.")
            login_signup_page()
        else:
            role = st.session_state["user"]["role"]
            if role == "admin":
                admin_dashboard()
            elif role == "teacher":
                teacher_dashboard()
            else:
                student_dashboard()
    elif choice == "Register Student & Capture Face":
        register_student_page()
    elif choice == "Train / Load Encodings":
        train_encodings_page()
    elif choice == "Mark Attendance (Camera)":
        mark_attendance_camera_page()
    elif choice == "Manual Attendance":
        manual_attendance_page()
    elif choice == "Attendance Reports":
        attendance_reports_page()
    elif choice == "Admin Analytics":
        admin_analytics_dashboard()


# ------------------------------ LOGIN + SIGNUP ------------------------------

def login_signup_page():
    st.subheader("Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    # Login
    with tab1:
        st.markdown("#### Login")
        uname = st.text_input("Username (Roll No for students)")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login_user(uname, pwd)
            if user:
                st.session_state["user"] = user
                st.session_state["is_authenticated"] = True
                st.session_state["current_page"] = "Dashboard"
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    # Signup
    with tab2:
        st.markdown("#### Create Account")
        full_name = st.text_input("Full Name")
        new_username = st.text_input("New Username (Student Roll No = Username)")
        new_password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["teacher", "admin", "student"])

        if st.button("Create Account"):
            success = register_new_user(new_username, new_password, full_name, role)
            if success:
                st.success("Account created. Please login.")
            else:
                st.error("Username already exists")


# ------------------------------ DASHBOARDS ------------------------------

def admin_dashboard():
    require_role(["admin"])
    user = st.session_state["user"]

    st.subheader(f"Admin Dashboard – {user['full_name']}")

    df = get_attendance_dataframe()

    students = get_students(None, None)

    total_students = len(students)
    total_records = len(df) if not df.empty else 0
    present_count = (df["status"] == "Present").sum() if not df.empty else 0
    absent_count = (df["status"] == "Absent").sum() if not df.empty else 0

    total_days = present_count + absent_count
    avg_attendance = f"{(present_count / total_days * 100):.1f}%" if total_days > 0 else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", total_students)
    c2.metric("Attendance Records", total_records)
    c3.metric("Present", present_count)
    c4.metric("Average Attendance", avg_attendance)

    section_divider()


def teacher_dashboard():
    require_role(["teacher"])
    user = st.session_state["user"]

    st.subheader(f"Teacher Dashboard – {user['full_name']}")
    st.info("Use sidebar options to manage attendance and students.")


def student_dashboard():
    require_role(["student"])
    user = st.session_state["user"]

    st.subheader(f"Student Dashboard – {user['full_name']}")

    # Auto-link student by username = roll_no
    student = get_student_by_username(user["username"])

    if student is None:
        st.error("No student profile found. Your username must be your roll number.")
        return

    sid = student["id"]

    st.markdown("#### Profile")
    st.json({
        "Roll No": student["student_id"],
        "Name": student["name"],
        "Class": student["class"],
        "Section": student["section"],
        "Email": student["email"],
    })

    df = attendance_to_dataframe(student_id=sid)
    if df.empty:
        st.info("No attendance records yet.")
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


# ------------------------------ ADMIN ANALYTICS ------------------------------

def admin_analytics_dashboard():
    require_role(["admin"])
    st.subheader("Admin Analytics")

    df = get_attendance_dataframe()
    if df.empty:
        st.info("No attendance data.")
        return

    df["date"] = pd.to_datetime(df["date"])

    # KPIs
    present_count = (df["status"] == "Present").sum()
    absent_count = (df["status"] == "Absent").sum()
    total = len(df)
    avg_attendance = f"{present_count / total * 100:.1f}%" if total > 0 else "N/A"

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", total)
    c2.metric("Present Count", present_count)
    c3.metric("Average Attendance", avg_attendance)

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


# ------------------------------ STUDENT REGISTRATION ------------------------------

def register_student_page():
    require_role(["admin", "teacher"])
    st.subheader("Register Student & Capture Face")

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

    st.markdown("### Capture Face")

    sid = st.session_state.get("last_student_id")
    if not sid:
        st.info("Register student first.")
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


# ------------------------------ TRAIN ENCODINGS ------------------------------

def train_encodings_page():
    require_role(["admin", "teacher"])
    st.subheader("Load Face Encodings")

    if st.button("Load Encodings"):
        ids, names, encs = load_known_face_encodings()
        st.session_state["known_ids"] = ids
        st.session_state["known_names"] = names
        st.session_state["known_encodings"] = encs

        st.success(f"Loaded {len(ids)} encodings.")


# ------------------------------ CAMERA ATTENDANCE ------------------------------

def mark_attendance_camera_page():
    require_role(["admin", "teacher"])
    st.subheader("Mark Attendance (Camera)")

    known_ids = st.session_state.get("known_ids", [])
    known_encs = st.session_state.get("known_encodings", [])
    known_names = st.session_state.get("known_names", [])

    if not known_encs:
        st.warning("Load encodings first.")
        return

    if not st.button("Start Camera"):
        st.info("Press Start Camera.")
        return

    frame_box = st.empty()
    log_box = st.empty()
    logs = []
    marked_today = set()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    for i in range(300):
        ret, frame = cap.read()
        if not ret:
            continue

        results = recognize_faces_in_frame(frame, known_encs, known_ids, known_names)

        new_ids, new_logs = mark_attendance_from_results(
            results,
            st.session_state["user"]["id"],
            marked_today
        )
        logs.extend(new_logs)

        frame = draw_face_boxes(frame, results)
        frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if logs:
            log_box.write("### Logs")
            for l in logs[-10:]:
                log_box.write(l)

        time.sleep(0.05)

        if st.button("Stop"):
            break

    cap.release()


# ------------------------------ MANUAL ATTENDANCE ------------------------------

def manual_attendance_page():
    require_role(["admin", "teacher"])
    st.subheader("Manual Attendance")

    cls = st.text_input("Class")
    sec = st.text_input("Section")
    date = st.date_input("Date").strftime("%Y-%m-%d")

    students = get_students(cls or None, sec or None)

    with st.form("manual_form"):
        status_map = {}

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
            if stv != "Not Set":
                mark_manual_attendance(
                    sid,
                    stv,
                    date,
                    st.session_state["user"]["id"]
                )
                count += 1

        st.success(f"Saved {count} attendance records.")


# ------------------------------ ATTENDANCE REPORTS ------------------------------

def attendance_reports_page():
    require_login()
    st.subheader("Attendance Reports")

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
            return

        st.dataframe(df)

        summ = calculate_attendance_summary(df)
        st.write("Summary")
        st.dataframe(summ)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "attendance.csv")


# ------------------------------ RUN APP ------------------------------

if __name__ == "__main__":
    main()
