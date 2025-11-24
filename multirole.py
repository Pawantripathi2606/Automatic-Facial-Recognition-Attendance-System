# roles_advanced.py
"""
Advanced Multi-Role Permission System
-------------------------------------

This module adds additional roles and maps them to the features
they are allowed to use. This keeps the app clean, scalable,
and easy to maintain.

New roles added:
- principal
- hr
- supervisor
- auditor
- guest
"""

import streamlit as st


# -------------------------------------------------------
# ROLE DEFINITIONS
# -------------------------------------------------------

ALL_ROLES = [
    "admin",
    "teacher",
    "student",
    "principal",
    "hr",
    "supervisor",
    "auditor",
    "guest",
]

# Page → allowed roles
ROLE_PAGE_ACCESS = {
    "Dashboard": ["admin", "teacher", "student", "principal", "supervisor", "guest"],

    "Mark Attendance (Camera)": ["admin", "teacher", "supervisor"],
    "Register Student & Capture Face": ["admin", "teacher"],
    "Train / Load Encodings": ["admin", "teacher"],

    "Manual Attendance": ["admin", "teacher"],
    "Attendance Reports": ["admin", "teacher", "principal", "hr", "student"],

    "Student Management": ["admin", "teacher"],
    "Attendance Management": ["admin", "principal", "hr"],

    "Admin Analytics": ["admin", "principal"],
    "Insights & Alerts": ["admin", "principal"],
    "Live Admin Monitor": ["admin", "principal", "supervisor"],

    "Database Control Center": ["admin", "principal"],
    "Timetable & Period-wise Attendance": ["admin", "teacher", "supervisor"],

    # Student Portal
    "Student Self-Service Portal": ["student"],

    # Future features can be added here
}


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def get_allowed_pages_for_role(role: str) -> list:
    """
    Returns a list of pages the user is allowed to see.
    """
    pages = []
    for page, roles in ROLE_PAGE_ACCESS.items():
        if role in roles:
            pages.append(page)
    return pages


def require_roles(allowed_roles: list):
    """
    Checks if logged in user role exists inside allowed_roles.
    If not, blocks the page.
    """
    current_user = st.session_state.get("user")
    if not current_user:
        st.error("You must be logged in.")
        st.stop()

    role = current_user["role"]

    if role not in allowed_roles:
        st.error(f"⛔ You do not have permission to access this page (role: {role}).")
        st.stop()


def restrict_page_access(page_name: str):
    """
    Call inside each routed page to automatically verify access.
    Example: restrict_page_access("Admin Analytics")
    """
    user = st.session_state.get("user")

    if not user:
        st.error("You must be logged in.")
        st.stop()

    role = user["role"]

    allowed_roles = ROLE_PAGE_ACCESS.get(page_name, [])

    if role not in allowed_roles:
        st.error("🚫 Access denied for this role.")
        st.stop()

