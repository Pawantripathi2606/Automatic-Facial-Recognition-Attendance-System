# ui_quick_actions.py

import streamlit as st
import pandas as pd
from datetime import datetime

from db import get_students
from attendance_utils import attendance_to_dataframe
from face_utils import load_known_face_encodings


def render_quick_actions_panel():
    """
    Small right-side panel with quick buttons:
    - Open camera attendance page (just hint)
    - Load encodings
    - Add student (scrolls)
    - Export today's report
    - Mark all present today (manual bulk)
    """
    st.markdown("### ⚙️ Quick Actions")

    # Row of small buttons
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    # 1) Go to camera page (hint UI – actual navigation user karega sidebar se)
    if c1.button("📸 Open Camera"):
        st.info("Use left sidebar → 'Mark Attendance (Camera)' to start the camera module.")

    # 2) Load encodings
    if c2.button("🧠 Load Encodings"):
        ids, names, encs = load_known_face_encodings()
        st.session_state["known_ids"] = ids
        st.session_state["known_names"] = names
        st.session_state["known_encodings"] = encs
        st.success(f"Loaded {len(ids)} encodings into memory.")

    # 3) Add student (scroll hint)
    if c3.button("➕ Add Student"):
        st.info("Go to 'Register Student & Capture Face' from sidebar to add a new student.")

    # 4) Export today's attendance
    if c4.button("⬇️ Export Today Report"):
        today = datetime.now().strftime("%Y-%m-%d")
        df = attendance_to_dataframe(None, None, None, today, today)
        if df.empty:
            st.warning("No attendance records for today.")
        else:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Today's CSV",
                csv,
                file_name=f"attendance_{today}.csv",
                mime="text/csv",
            )

    st.markdown("---")

    # Optional: quick stats for today
    today = datetime.now().strftime("%Y-%m-%d")
    df_today = attendance_to_dataframe(None, None, None, today, today)
    students = get_students(None, None)
    total_students = len(students)
    present = (df_today["status"] == "Present").sum() if not df_today.empty else 0

    st.markdown("#### 📅 Today at a glance")
    st.write(
        f"**Present:** {present} / {total_students}  \n"
        f"**Records logged:** {len(df_today) if not df_today.empty else 0}"
    )
