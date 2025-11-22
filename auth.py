# auth.py

import bcrypt
from typing import Optional
from db import get_user_by_username, create_user, count_users
import streamlit as st

# ------------- Password hashing ------------- #

def hash_password(password: str) -> str:
    """Hash the password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


# ------------- User registration / login ------------- #

def register_new_user(username: str, password: str, full_name: str, role: str) -> bool:
    """Register user (first user always admin)."""
    users_count = count_users()
    if users_count == 0:
        role = "admin"

    password_hash = hash_password(password)
    return create_user(username, password_hash, full_name, role)


def login_user(username: str, password: str) -> Optional[dict]:
    """Login validation."""
    user_row = get_user_by_username(username)
    if not user_row:
        return None
    if not verify_password(password, user_row["password_hash"]):
        return None

    return {
        "id": user_row["id"],
        "username": user_row["username"],
        "full_name": user_row["full_name"],
        "role": user_row["role"],
    }


# ----------- FIXED SESSION STATE ----------- #

def init_session_state():
    """Safe initialization of session state."""
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "is_authenticated" not in st.session_state:
        st.session_state["is_authenticated"] = False


def logout_user():
    """Logout user safely."""
    st.session_state["user"] = None
    st.session_state["is_authenticated"] = False


# ----------- PAGE GUARDS ----------- #

def require_login():
    """Ensure user logged in."""
    if not st.session_state.get("is_authenticated"):
        st.warning("Please login to access this page.")
        st.stop()


def require_role(allowed_roles):
    """Ensure required role."""
    require_login()
    user_role = st.session_state["user"]["role"]
    if user_role not in allowed_roles:
        st.error("Access denied.")
        st.stop()
