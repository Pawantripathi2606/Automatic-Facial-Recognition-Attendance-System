# Automatic Facial Recognition Attendance System.

A complete, end-to-end attendance automation system built using:

- Python  
- Streamlit  
- InsightFace (ArcFace)  
- OpenCV  
- SQLite  
- ONNX Runtime  

This application performs real-time face recognition using ArcFace embeddings, stores attendance, and provides dashboards for Admin, Teacher, and Student roles.

---

## Features

### Authentication
- Secure login (bcrypt hashed passwords)
- Roles: Admin, Teacher, Student
- Student login username = Student Roll Number

### Student Management
- Register new students  
- Capture face using Streamlit camera  
- Save 512D ArcFace face embeddings  

### Attendance System
- Real-time face recognition via webcam  
- Auto-mark attendance once per day  
- Manual attendance marking  

### Dashboards & Analytics
- Role-based dashboards  
- Student dashboard with profile + monthly chart  
- Admin analytics:
  - Daily attendance trend  
  - Class/section-based summary  
  - Present vs Absent distribution  
- Attendance reports with CSV export  

---

## Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **Database**: SQLite  
- **Face Recognition**: InsightFace ArcFace (ONNX Runtime)  
- **Charts**: Matplotlib, Seaborn  

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Pawantripathi2606/Automatic-Facial-Recognition-Attendance-System.git
cd Automatic-Facial-Recognition-Attendance-System

#create virtual enviroment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux


pip install -r requirements.txt


streamlit run app.py

# project structure

face_reco_sys/
│── app.py                    # Main Streamlit application
│── auth.py                   # Login / Signup logic
│── db.py                     # SQLite database functions
│── face_utils.py             # ArcFace detection + encoding + recognition
│── attendance_utils.py       # Reports, analytics, manual attendance
│── config.py                 # Config constants
│── requirements.txt          # Python dependencies
│── runtime.txt               # Required for Render (Python runtime)
│── .gitignore
└── attendance_system.db      # Local SQLite DB (ignored from Git)


