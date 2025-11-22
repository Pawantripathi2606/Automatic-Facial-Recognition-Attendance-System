# attendance_utils.py

import sqlite3
import pandas as pd
from db import get_connection


# ---------------------------------------------------------
# Helper: Convert attendance DB rows into a pandas DataFrame
# ---------------------------------------------------------

def attendance_to_dataframe(student_id=None, class_name=None, section=None,
                            start_date=None, end_date=None):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT 
            attendance.id,
            attendance.student_id,
            students.student_id AS roll_no,
            students.name,
            students.class,
            students.section,
            attendance.date,
            attendance.time,
            attendance.status,
            attendance.marked_by
        FROM attendance
        JOIN students ON students.id = attendance.student_id
        WHERE 1=1
    """

    params = []

    if student_id:
        query += " AND attendance.student_id = ?"
        params.append(student_id)

    if class_name:
        query += " AND students.class = ?"
        params.append(class_name)

    if section:
        query += " AND students.section = ?"
        params.append(section)

    if start_date:
        query += " AND attendance.date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND attendance.date <= ?"
        params.append(end_date)

    query += " ORDER BY attendance.date DESC, attendance.time DESC"

    rows = cursor.execute(query, params).fetchall()
    cols = [desc[0] for desc in cursor.description]

    conn.close()

    if not rows:
        return pd.DataFrame(columns=[
            "id", "student_id", "roll_no", "name", "class", "section",
            "date", "time", "status", "marked_by"
        ])

    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------
# Used by admin dashboard & analytics
# ---------------------------------------------------------

def get_attendance_dataframe():
    conn = get_connection()
    query = """
        SELECT 
            attendance.id,
            attendance.student_id,
            students.student_id AS roll_no,
            students.name,
            students.class,
            students.section,
            attendance.date,
            attendance.time,
            attendance.status,
            attendance.marked_by
        FROM attendance
        JOIN students ON students.id = attendance.student_id
        ORDER BY attendance.date DESC, attendance.time DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ---------------------------------------------------------
# Summary table used in reports
# ---------------------------------------------------------

def calculate_attendance_summary(df):
    if df.empty:
        return pd.DataFrame()

    present = (df["status"] == "Present").sum()
    absent = (df["status"] == "Absent").sum()
    total = len(df)

    percent = (present / total * 100) if total > 0 else 0

    summary = {
        "Total Days": [total],
        "Present": [present],
        "Absent": [absent],
        "Attendance %": [round(percent, 2)],
    }

    return pd.DataFrame(summary)


# ---------------------------------------------------------
# Student Monthly Attendance Graph (bar chart)
# ---------------------------------------------------------

def get_student_monthly_graph(df, student_id):
    if df.empty:
        return pd.DataFrame()

    df = df[df["status"] == "Present"].copy()
    if df.empty:
        return pd.DataFrame()

    df["month"] = pd.to_datetime(df["date"]).dt.strftime("%b %Y")

    monthly = df.groupby("month").size().reset_index(name="Present Days")
    return monthly


# ---------------------------------------------------------
# Class-wise attendance summary (used in Admin analytics)
# ---------------------------------------------------------

def get_class_wise_summary(df):
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby(["class", "section", "status"]).size().reset_index(name="count")
    return summary


# ---------------------------------------------------------
# Daily attendance trend
# ---------------------------------------------------------

def get_daily_attendance_summary(df):
    if df.empty:
        return pd.DataFrame()

    daily = (
        df[df["status"] == "Present"]
        .groupby("date")
        .size()
        .reset_index(name="Present Count")
    )

    return daily


# ---------------------------------------------------------
# Manual Attendance
# ---------------------------------------------------------

def mark_manual_attendance(student_id, status, date, marked_by):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO attendance (student_id, date, time, status, marked_by)
        VALUES (?, ?, time('now', 'localtime'), ?, ?)
        """,
        (student_id, date, status, marked_by),
    )

    conn.commit()
    conn.close()
