# timetable.py
"""
Timetable + Period-wise Attendance Module
-----------------------------------------
Features:
- Maintain class timetables
- Track attendance by period (1st period, 2nd period, lab, etc.)
- Admin/Teacher can add/update/delete timetable slots
- Students' attendance can be filtered period-wise
"""

import streamlit as st
import sqlite3
from datetime import datetime
from db import get_connection, get_students
from attendance_utils import mark_manual_attendance, attendance_to_dataframe


# ----------------------------------------------------
# DATABASE INITIALIZATION FOR TIMETABLE
# ----------------------------------------------------

def init_timetable_table():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS timetable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class TEXT NOT NULL,
            section TEXT NOT NULL,
            period TEXT NOT NULL,
            subject TEXT NOT NULL,
            teacher TEXT,
            start_time TEXT,
            end_time TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS period_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            class TEXT NOT NULL,
            section TEXT NOT NULL,
            period TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            marked_by INTEGER
        )
    """)

    conn.commit()
    conn.close()


# ----------------------------------------------------
# CRUD FOR TIMETABLE
# ----------------------------------------------------

def add_period(class_, section, period, subject, teacher, start, end):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO timetable (class, section, period, subject, teacher, start_time, end_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (class_, section, period, subject, teacher, start, end))
    conn.commit()
    conn.close()


def get_timetable(class_=None, section=None):
    conn = get_connection()
    cur = conn.cursor()

    query = "SELECT * FROM timetable WHERE 1=1"
    params = []

    if class_:
        query += " AND class = ?"
        params.append(class_)

    if section:
        query += " AND section = ?"
        params.append(section)

    query += " ORDER BY id ASC"
    cur.execute(query, params)
    rows = cur.fetchall()

    conn.close()
    return rows


def delete_period(period_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM timetable WHERE id = ?", (period_id,))
    conn.commit()
    conn.close()


# ----------------------------------------------------
# PERIOD-WISE ATTENDANCE
# ----------------------------------------------------

def mark_period_attendance(student_id, class_, section, period, status, marked_by):
    conn = get_connection()
    cur = conn.cursor()

    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")

    cur.execute("""
        INSERT INTO period_attendance
        (student_id, class, section, period, date, time, status, marked_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (student_id, class_, section, period, date, time, status, marked_by))

    conn.commit()
    conn.close()


def get_period_attendance(class_=None, section=None, period=None, date=None):
    conn = get_connection()
    cur = conn.cursor()

    query = "SELECT * FROM period_attendance WHERE 1=1"
    params = []

    if class_:
        query += " AND class = ?"
        params.append(class_)

    if section:
        query += " AND section = ?"
        params.append(section)

    if period:
        query += " AND period = ?"
        params.append(period)

    if date:
        query += " AND date = ?"
        params.append(date)

    query += " ORDER BY date DESC, time DESC"

    cur.execute(query, params)
    rows = cur.fetchall()

    conn.close()
    return rows


# ----------------------------------------------------
# TIMETABLE UI PAGE (ADMIN + TEACHER)
# ----------------------------------------------------

def timetable_page():
    st.subheader("📘 Timetable & Period-wise Attendance")

    st.markdown("### 1️⃣ Add New Period")
    class_ = st.text_input("Class")
    section = st.text_input("Section")
    period = st.selectbox("Period", ["1st", "2nd", "3rd", "4th", "5th", "Lab", "Activity"])
    subject = st.text_input("Subject")
    teacher = st.text_input("Teacher Name")
    start = st.time_input("Start Time").strftime("%H:%M")
    end = st.time_input("End Time").strftime("%H:%M")

    if st.button("Add Period"):
        add_period(class_, section, period, subject, teacher, start, end)
        st.success("Period added successfully!")

    st.markdown("---")

    st.markdown("### 2️⃣ View Timetable")
    class_view = st.text_input("Filter Class")
    section_view = st.text_input("Filter Section")

    table = get_timetable(class_view, section_view)
    if table:
        for row in table:
            with st.expander(f"{row['period']} – {row['subject']}"):
                st.write(dict(row))

                if st.button("Delete", key=f"del_{row['id']}"):
                    delete_period(row["id"])
                    st.warning("Period deleted.")
                    st.rerun()
    else:
        st.info("No timetable found.")

    st.markdown("---")

    st.markdown("### 3️⃣ Mark Period-wise Attendance")
    cls = st.text_input("Class for attendance")
    sec = st.text_input("Section for attendance")
    per = st.selectbox("Select Period", ["1st", "2nd", "3rd", "4th", "5th", "Lab", "Activity"])

    students = get_students(cls or None, sec or None)

    if not students:
        st.info("No students found.")
        return

    with st.form("period_attendance_form"):
        status_map = {}
        for s in students:
            status = st.selectbox(
                f"{s['student_id']} - {s['name']}",
                ["Present", "Absent"],
                key=f"p_{s['id']}",
            )
            status_map[s["id"]] = status

        submit = st.form_submit_button("Save Attendance")

    if submit:
        for sid, stv in status_map.items():
            mark_period_attendance(
                sid,
                cls,
                sec,
                per,
                stv,
                st.session_state["user"]["id"],
            )
        st.success("Period-wise attendance saved successfully!")

    st.markdown("---")

    st.markdown("### 4️⃣ View Period Attendance Logs")
    view_date = st.date_input("Date").strftime("%Y-%m-%d")
    logs = get_period_attendance(cls, sec, per, view_date)

    if logs:
        st.write(pd.DataFrame(logs))
    else:
        st.info("No records found for selected filters.")

