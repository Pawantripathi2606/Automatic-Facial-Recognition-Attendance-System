# student_portal.py
"""
Student Self-Service Portal
---------------------------
Features:
- Student dashboard
- Attendance heatmap
- Attendance trend chart
- Download report
- Request correction
- Update profile picture (admin approval required)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from attendance_utils import attendance_to_dataframe
from face_utils import encode_single_face_from_frame, save_student_face_encoding
from heatmap_utils import generate_empty_seat_map, map_attendance_to_seats, draw_heatmap
from db import get_student_by_username, update_student


sns.set_style("whitegrid")


# -------------------------------------------------------------------
# MAIN STUDENT PORTAL PAGE
# -------------------------------------------------------------------

def student_portal_page():
    """
    Main Student Portal UI – students can view their own complete data.
    """
    st.subheader("🎓 Student Self-Service Portal")

    # Validate logged-in student
    user = st.session_state.get("user", None)
    if not user or user["role"] != "student":
        st.error("This section is only for students.")
        st.stop()

    stu = get_student_by_username(user["username"])
    if not stu:
        st.error("Student profile not found. Your username must be your Roll Number.")
        st.stop()

    student_id = stu["id"]
    class_ = stu["class"]
    section = stu["section"]

    # Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📘 Dashboard", "📊 Attendance Trend", "🟩 Heatmap", "📝 Correction Request", "📷 Update Face"]
    )

    # ---------------------------------------------------------
    # TAB 1 – DASHBOARD
    # ---------------------------------------------------------
    with tab1:
        st.markdown("### 👤 Profile")
        st.json({
            "Roll No": stu["student_id"],
            "Name": stu["name"],
            "Class": class_,
            "Section": section,
            "Email": stu["email"]
        })

        df = attendance_to_dataframe(student_id=student_id)
        if df.empty:
            st.info("No attendance records yet.")
        else:
            st.markdown("---")
            st.markdown("### 📈 Summary")

            total_days = len(df)
            present = (df["status"] == "Present").sum()
            absent = (df["status"] == "Absent").sum()
            percent = (present / total_days * 100) if total_days else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Days", total_days)
            c2.metric("Present", present)
            c3.metric("Attendance %", f"{percent:.1f}%")

            st.markdown("---")
            st.markdown("### 📥 Download Full Report")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV Report",
                csv,
                file_name=f"{stu['student_id']}_attendance.csv",
                mime="text/csv"
            )

    # ---------------------------------------------------------
    # TAB 2 – ATTENDANCE TREND
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### 📊 Monthly Attendance Trend")

        df = attendance_to_dataframe(student_id=student_id)
        if df.empty:
            st.info("No data found.")
        else:
            df["date"] = pd.to_datetime(df["date"])
            df["month"] = df["date"].dt.to_period("M")

            monthly = (df[df["status"] == "Present"]
                       .groupby("month")
                       .size()
                       .reset_index(name="Present Days"))

            fig, ax = plt.subplots(figsize=(7, 3))
            sns.barplot(data=monthly, x="month", y="Present Days", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # ---------------------------------------------------------
    # TAB 3 – HEATMAP
    # ---------------------------------------------------------
    with tab3:
        st.markdown("### 🟩 Classroom Heatmap (Your Attendance Position)")

        rows = st.slider("Rows", 3, 10, 5)
        cols = st.slider("Columns", 3, 10, 6)

        date = st.date_input("Select Date").strftime("%Y-%m-%d")

        if st.button("Generate Heatmap"):
            full_df = attendance_to_dataframe(student_id=None, class_=class_, section=section,
                                              start_date=date, end_date=date)

            if full_df.empty:
                st.warning("No attendance data for selected date.")
            else:
                seat_map = generate_empty_seat_map(rows, cols)
                seat_map = map_attendance_to_seats(full_df, seat_map)
                draw_heatmap(seat_map)

    # ---------------------------------------------------------
    # TAB 4 – ATTENDANCE CORRECTION REQUEST
    # ---------------------------------------------------------
    with tab4:
        st.markdown("### 📝 Attendance Correction Request")

        date = st.date_input("Select Date to Correct").strftime("%Y-%m-%d")
        reason = st.text_area("Reason for correction")

        if st.button("Submit Correction Request"):
            if not reason.strip():
                st.error("Please provide a reason.")
            else:
                # Log request (store in a simple file)
                with open("correction_requests.txt", "a") as f:
                    f.write(f"{student_id},{stu['student_id']},{stu['name']},{date},{reason}\n")

                st.success("Your correction request has been submitted. Admin will review it.")

    # ---------------------------------------------------------
    # TAB 5 – UPDATE FACE
    # ---------------------------------------------------------
    with tab5:
        st.markdown("### 📷 Update Your Face Image")

        st.info("Your updated face will be saved and admin must approve it.")

        img = st.camera_input("Capture New Face")
        if img:
            data = np.frombuffer(img.read(), np.uint8)
            frame = cv2.imdecode(data, 1)

            encoding = encode_single_face_from_frame(frame)
            if encoding is None:
                st.error("No face detected. Try again.")
            else:
                save_student_face_encoding(student_id, encoding)
                st.success("Face updated successfully! Admin review pending.")
