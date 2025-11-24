# heatmap_utils.py
"""
Attendance Heatmap Module
-------------------------
This module generates a classroom heatmap showing:
- Present = Green
- Absent = Red
- No record = Gray

Supports period-wise and date-wise filters.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from attendance_utils import attendance_to_dataframe
from timetable import get_period_attendance


# -------------------------------------------------------
# CONFIG – Classroom layout (editable)
# -------------------------------------------------------
DEFAULT_ROWS = 5       # 5 rows
DEFAULT_COLS = 6       # 6 seats per row


def generate_empty_seat_map(rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
    """
    Generates an empty 2D matrix for seat map.
    """
    return np.full((rows, cols), -1)   # -1 = no data


# -------------------------------------------------------
# Mapping attendance to seat-map
# -------------------------------------------------------

def map_attendance_to_seats(df, seat_map):
    """
    Takes attendance dataframe + seat map template,
    returns seat map with:
    1 = Present
    0 = Absent
    -1 = No record
    """
    updated_map = seat_map.copy()
    rows, cols = updated_map.shape

    # Ensure df has 1 row per student
    df_unique = df.groupby("student_id").first().reset_index()

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(df_unique):
                updated_map[r, c] = -1
                continue

            status = df_unique.iloc[idx]["status"]
            if status == "Present":
                updated_map[r, c] = 1
            elif status == "Absent":
                updated_map[r, c] = 0
            else:
                updated_map[r, c] = -1

            idx += 1

    return updated_map


# -------------------------------------------------------
# HEATMAP RENDERING
# -------------------------------------------------------

def draw_heatmap(seat_map):
    """
    Visual heatmap rendering.
    Colors:
    Green = Present
    Red = Absent
    Gray = No record
    """
    rows, cols = seat_map.shape

    # Color mapping
    colors = {
        1: "#16a34a",   # green (present)
        0: "#dc2626",   # red (absent)
        -1: "#94a3b8",  # gray (no record)
    }

    fig, ax = plt.subplots(figsize=(cols, rows))
    for r in range(rows):
        for c in range(cols):
            value = seat_map[r, c]
            rect = plt.Rectangle(
                (c, rows - r - 1),
                1,
                1,
                facecolor=colors[value],
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])

    st.pyplot(fig)


# -------------------------------------------------------
# MAIN UI PAGE
# -------------------------------------------------------

def heatmap_page():
    """
    Main heatmap page, for admins & teachers.
    """
    st.subheader("🟩 Classroom Attendance Heatmap")

    st.markdown(
        "Visualize classroom attendance as a seat-map. "
        "Green = Present, Red = Absent, Gray = No record"
    )

    class_ = st.text_input("Class")
    section = st.text_input("Section")
    date = st.date_input("Date").strftime("%Y-%m-%d")
    period = st.text_input("Period (Optional)", placeholder="1st, 2nd, Lab, etc.")

    rows = st.number_input("Rows", 1, 20, DEFAULT_ROWS)
    cols = st.number_input("Columns", 1, 20, DEFAULT_COLS)

    if st.button("Generate Heatmap"):
        seat_map = generate_empty_seat_map(rows, cols)

        # CASE 1: Period-wise attendance
        if period.strip():
            logs = get_period_attendance(class_, section, period, date)
            if not logs:
                st.warning("No period-attendance found for selected filters.")
                seat_map[:] = -1
            else:
                df = pd.DataFrame(logs)
                seat_map = map_attendance_to_seats(df, seat_map)

        # CASE 2: Daily attendance
        else:
            df = attendance_to_dataframe(
                student_id=None,
                class_=class_,
                section=section,
                start_date=date,
                end_date=date,
            )

            if df.empty:
                st.warning("No attendance data found for selected class.")
                seat_map[:] = -1
            else:
                seat_map = map_attendance_to_seats(df, seat_map)

        draw_heatmap(seat_map)

    st.info("Tip: Register students in correct seating order for consistent heatmap mapping.")
