import sqlite3
import json
from config import DB_PATH


# ------------------------- CONNECTION ------------------------- #

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------- INIT DB ------------------------- #

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # Students table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,  -- roll number
            name TEXT NOT NULL,
            class TEXT,
            section TEXT,
            email TEXT,
            face_encoding TEXT               -- JSON encoded embedding
        )
    """)

    # Attendance table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            marked_by INTEGER,
            FOREIGN KEY(student_id) REFERENCES students(id),
            FOREIGN KEY(marked_by) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# ------------------------- USERS ------------------------- #

def get_user_by_username(username: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row


def create_user(username, password_hash, full_name, role) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
            (username, password_hash, full_name, role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def count_users() -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    n = cur.fetchone()[0]
    conn.close()
    return n


# ------------------------- STUDENTS ------------------------- #

def create_student(student_id, name, cls, sec, email):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO students (student_id, name, class, section, email)
            VALUES (?, ?, ?, ?, ?)
            """,
            (student_id, name, cls, sec, email),
        )
        conn.commit()
        return True, cur.lastrowid
    except sqlite3.IntegrityError:
        return False, None
    finally:
        conn.close()


def get_students(cls=None, sec=None):
    conn = get_connection()
    cur = conn.cursor()

    query = "SELECT * FROM students WHERE 1=1"
    params = []

    if cls:
        query += " AND class = ?"
        params.append(cls)
    if sec:
        query += " AND section = ?"
        params.append(sec)

    query += " ORDER BY class, section, student_id"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_student_by_student_id(roll_no: str):
    """Find student profile by roll number (used for reports)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE student_id = ?", (roll_no,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_student_by_username(username: str):
    """
    Auto-link students by login username = roll number.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE student_id = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_students_with_encodings():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, face_encoding FROM students WHERE face_encoding IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    return rows


def update_student_face_encoding(student_id: int, encoding_json: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE students SET face_encoding = ? WHERE id = ?",
        (encoding_json, student_id),
    )
    conn.commit()
    conn.close()


# NEW: update student basic details (admin/teacher edit)
def update_student(student_pk: int, student_id: str, name: str, cls: str, sec: str, email: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE students
        SET student_id = ?, name = ?, class = ?, section = ?, email = ?
        WHERE id = ?
        """,
        (student_id, name, cls, sec, email, student_pk),
    )
    conn.commit()
    conn.close()


# NEW: delete student (and optionally related attendance)
def delete_student(student_pk: int, delete_attendance: bool = False):
    conn = get_connection()
    cur = conn.cursor()
    if delete_attendance:
        cur.execute("DELETE FROM attendance WHERE student_id = ?", (student_pk,))
    cur.execute("DELETE FROM students WHERE id = ?", (student_pk,))
    conn.commit()
    conn.close()


# ------------------------- ATTENDANCE ------------------------- #

def insert_attendance(student_id, date, time, status, marked_by):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO attendance (student_id, date, time, status, marked_by)
        VALUES (?, ?, ?, ?, ?)
        """,
        (student_id, date, time, status, marked_by),
    )
    conn.commit()
    conn.close()


def has_attendance_for_date(student_id, date):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?",
        (student_id, date),
    )
    n = cur.fetchone()[0]
    conn.close()
    return n > 0


def get_all_attendance_records():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
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
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# NEW: update attendance status (admin manual edit)
def update_attendance_status(att_id: int, status: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE attendance SET status = ? WHERE id = ?",
        (status, att_id),
    )
    conn.commit()
    conn.close()


# NEW: delete a single attendance record
def delete_attendance_record(att_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM attendance WHERE id = ?", (att_id,))
    conn.commit()
    conn.close()


# ------------------------- BULK DELETE (SAFE MODE) ------------------------- #

def delete_all_attendance():
    """Delete ALL attendance rows, but keep table structure."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()


def clear_all_face_encodings():
    """Remove all face encodings but keep student records."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE students SET face_encoding = NULL")
    conn.commit()
    conn.close()
