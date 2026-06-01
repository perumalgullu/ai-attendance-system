import os
import cv2
import pickle
import json
import csv
import tempfile
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine
from deepface import DeepFace

# ============================
# CONSTANTS
# ============================

EMBED_FILE = "student_embeddings.pkl"
CONTACT_FILE = "contacts.json"
COSINE_THRESHOLD = 0.40
DB_FOLDER = "student_database"

# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="AI Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CUSTOM CSS
# ============================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1f3cff 0%, #6c5ce7 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #1f3cff;
    }

    .present-badge {
        background: #e8f8f0;
        color: #00b894;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .absent-badge {
        background: #fdecea;
        color: #e17055;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .student-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# CONTACT STORAGE
# ============================

def load_contacts():
    if os.path.exists(CONTACT_FILE):
        with open(CONTACT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_contacts(data):
    with open(CONTACT_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ============================
# FACE UTILS
# ============================

def get_face_embedding(image_path):
    try:
        reps = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        return reps[0]["embedding"]
    except Exception:
        return None

def detect_and_embed_faces(image_path):
    try:
        reps = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        return [r["embedding"] for r in reps]
    except Exception:
        return []

# ============================
# MAIN APP
# ============================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 AI Face Recognition Attendance System</h1>
        <p style="margin:0; opacity:0.85;">Powered by DeepFace · ArcFace · RetinaFace</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/student-male.png", width=60)
        st.markdown("## Navigation")
        page = st.radio("", [
            "➕ Add Student",
            "📁 Enroll Students",
            "📸 Mark Attendance",
            "📜 View History",
            "👤 Student Profile"
        ], label_visibility="collapsed")

        st.divider()

        # Quick stats
        contacts = load_contacts()
        st.markdown(f"**👥 Registered Students:** {len(contacts)}")

        enrolled = os.path.exists(EMBED_FILE)
        if enrolled:
            with open(EMBED_FILE, "rb") as f:
                known = pickle.load(f)
            st.markdown(f"**✅ Enrolled:** {len(known)} students")
        else:
            st.markdown("**⚠️ Not yet enrolled**")

        csv_files = [f for f in os.listdir() if f.startswith("attendance_") and f.endswith(".csv")]
        st.markdown(f"**📄 Sessions Recorded:** {len(csv_files)}")

    # ── Page routing ──────────────────────────────────────────
    if "➕ Add Student" in page:
        page_add_student()
    elif "📁 Enroll Students" in page:
        page_enroll_students()
    elif "📸 Mark Attendance" in page:
        page_mark_attendance()
    elif "📜 View History" in page:
        page_view_history()
    elif "👤 Student Profile" in page:
        page_student_profile()


# ============================
# PAGE: ADD STUDENT
# ============================

def page_add_student():
    st.header("➕ Add New Student")
    st.markdown("Register a new student with their photo and contact details.")

    col1, col2 = st.columns(2)

    with col1:
        roll = st.text_input("Roll Number", placeholder="e.g. 001")
        name = st.text_input("Student Name", placeholder="e.g. Perumal")

        if roll and name:
            sid = f"{roll.strip().zfill(3)}_{name.strip().lower().replace(' ', '_')}"
            st.info(f"📌 Student ID will be: **{sid}**")

    with col2:
        email = st.text_input("Email Address", placeholder="student@email.com")
        phone = st.text_input("Phone Number", placeholder="+91XXXXXXXXXX")

    photo = st.file_uploader("📷 Upload Student Photo", type=["jpg", "jpeg", "png"])

    if photo:
        st.image(photo, caption="Preview", width=200)

    if st.button("💾 Save Student", type="primary"):
        if not roll or not name:
            st.error("❌ Roll Number and Name are required")
        elif not photo:
            st.error("❌ Please upload a photo")
        else:
            student_id = f"{roll.strip().zfill(3)}_{name.strip().lower().replace(' ', '_')}"
            folder = os.path.join(DB_FOLDER, student_id)
            os.makedirs(folder, exist_ok=True)

            img = Image.open(photo)
            img.save(os.path.join(folder, photo.name))

            contacts = load_contacts()
            contacts[student_id] = {"email": email, "phone": phone}
            save_contacts(contacts)

            st.success(f"✅ Student **{student_id}** saved! Now go to **Enroll Students**.")
            st.balloons()


# ============================
# PAGE: ENROLL STUDENTS
# ============================

def page_enroll_students():
    st.header("📁 Enroll Students")
    st.markdown("Generate face embeddings from the student database folder.")

    if not os.path.exists(DB_FOLDER):
        st.error("❌ `student_database` folder not found. Add students first.")
        return

    student_dirs = [d for d in os.listdir(DB_FOLDER)
                    if os.path.isdir(os.path.join(DB_FOLDER, d))]

    if not student_dirs:
        st.warning("⚠️ No students found in the database folder.")
        return

    st.info(f"📂 Found **{len(student_dirs)}** student folders: `{', '.join(student_dirs)}`")

    if st.button("🚀 Start Enrollment", type="primary"):
        students = []
        progress = st.progress(0, text="Starting enrollment...")
        log = st.empty()

        for i, student_id in enumerate(student_dirs):
            path = os.path.join(DB_FOLDER, student_id)
            embeddings = []

            log.markdown(f"🔄 Processing **{student_id}**...")

            for img_file in os.listdir(path):
                ext = img_file.lower()
                if not (ext.endswith(".jpg") or ext.endswith(".png") or ext.endswith(".jpeg")):
                    continue
                emb = get_face_embedding(os.path.join(path, img_file))
                if emb is not None:
                    embeddings.append(emb)

            if embeddings:
                students.append({"id": student_id, "embeddings": embeddings})
                log.markdown(f"✅ **{student_id}** — {len(embeddings)} face(s) enrolled")
            else:
                log.markdown(f"⚠️ **{student_id}** — No face detected, skipped")

            progress.progress((i + 1) / len(student_dirs),
                              text=f"Enrolling {i+1}/{len(student_dirs)}...")

        with open(EMBED_FILE, "wb") as f:
            pickle.dump(students, f)

        progress.progress(1.0, text="✅ Enrollment complete!")
        st.success(f"🎉 Successfully enrolled **{len(students)}** students!")


# ============================
# PAGE: MARK ATTENDANCE
# ============================

def page_mark_attendance():
    st.header("📸 Mark Attendance")

    if not os.path.exists(EMBED_FILE):
        st.error("❌ Please enroll students first (go to Enroll Students).")
        return

    with open(EMBED_FILE, "rb") as f:
        known_students = pickle.load(f)

    all_ids = {s["id"] for s in known_students}

    st.markdown("Upload a class photo or multiple photos to mark attendance.")

    tab1, tab2 = st.tabs(["📤 Upload Photo(s)", "📊 Session Results"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload class photo(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded:
            present = set()

            for up in uploaded:
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(up, caption=up.name, use_container_width=True)

                with col2:
                    with st.spinner(f"Analysing {up.name}..."):
                        # Save to temp file for DeepFace
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            tmp.write(up.read())
                            tmp_path = tmp.name

                        detected = detect_and_embed_faces(tmp_path)
                        os.unlink(tmp_path)

                        found = []
                        for unknown in detected:
                            best = None
                            min_d = COSINE_THRESHOLD
                            for s in known_students:
                                for k in s["embeddings"]:
                                    d = cosine(unknown, k)
                                    if d < min_d:
                                        min_d = d
                                        best = s["id"]
                            if best:
                                present.add(best)
                                found.append(best)

                        if found:
                            st.success(f"✅ Recognised: **{', '.join(found)}**")
                        else:
                            st.warning("⚠️ No known faces detected in this photo.")

            # Store results in session state
            st.session_state["present"] = present
            st.session_state["absent"] = list(all_ids - present)
            st.session_state["session_done"] = True

    with tab2:
        if st.session_state.get("session_done"):
            present = st.session_state["present"]
            absent = st.session_state["absent"]

            col1, col2 = st.columns(2)
            col1.metric("✅ Present", len(present))
            col2.metric("❌ Absent", len(absent))

            st.divider()

            r1, r2 = st.columns(2)

            with r1:
                st.markdown("### ✅ Present")
                for p in sorted(present):
                    st.markdown(f'<span class="present-badge">✔ {p}</span>', unsafe_allow_html=True)
                    st.write("")

            with r2:
                st.markdown("### ❌ Absent")
                for a in sorted(absent):
                    st.markdown(f'<span class="absent-badge">✘ {a}</span>', unsafe_allow_html=True)
                    st.write("")

            if st.button("💾 Save Attendance CSV", type="primary"):
                fname = f"attendance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(fname, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["Student ID", "Status", "Time"])
                    for p in present:
                        w.writerow([p, "Present", now])
                    for a in absent:
                        w.writerow([a, "Absent", now])
                st.success(f"✅ Saved as `{fname}`")
        else:
            st.info("ℹ️ Upload photos in the first tab to see results here.")


# ============================
# PAGE: VIEW HISTORY
# ============================

def page_view_history():
    st.header("📜 Attendance History")

    csv_files = sorted(
        [f for f in os.listdir() if f.startswith("attendance_") and f.endswith(".csv")],
        reverse=True
    )

    if not csv_files:
        st.info("ℹ️ No attendance records found yet.")
        return

    selected = st.selectbox("Select a session", csv_files)

    if selected:
        rows = []
        with open(selected, newline="") as f:
            r = csv.reader(f)
            headers = next(r, None)
            for row in r:
                rows.append(row)

        present_count = sum(1 for r in rows if len(r) > 1 and r[1] == "Present")
        absent_count  = sum(1 for r in rows if len(r) > 1 and r[1] == "Absent")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Students", len(rows))
        c2.metric("✅ Present", present_count)
        c3.metric("❌ Absent", absent_count)

        st.divider()
        st.subheader(f"📋 Records — {selected}")

        for row in rows:
            if len(row) < 2:
                continue
            status_html = (
                f'<span class="present-badge">✔ Present</span>'
                if row[1] == "Present"
                else f'<span class="absent-badge">✘ Absent</span>'
            )
            col1, col2, col3 = st.columns([2, 1, 2])
            col1.markdown(f"**{row[0]}**")
            col2.markdown(status_html, unsafe_allow_html=True)
            col3.markdown(row[2] if len(row) > 2 else "")

        # Download button
        with open(selected, "rb") as f:
            st.download_button(
                label="⬇️ Download CSV",
                data=f,
                file_name=selected,
                mime="text/csv"
            )


# ============================
# PAGE: STUDENT PROFILE
# ============================

def page_student_profile():
    st.header("👤 Student Profile")

    contacts = load_contacts()

    if not contacts:
        st.info("ℹ️ No students registered yet.")
        return

    student_id = st.selectbox("Select Student", list(contacts.keys()))

    if not student_id:
        return

    info = contacts[student_id]

    # Count attendance
    present_days = set()
    absent_days = set()
    csv_files = [f for f in os.listdir() if f.startswith("attendance_") and f.endswith(".csv")]

    for fname in csv_files:
        with open(fname, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 3 and row[0] == student_id:
                    date_only = row[2].split(" ")[0]
                    if row[1] == "Present":
                        present_days.add(date_only)
                    elif row[1] == "Absent":
                        absent_days.add(date_only)

    total = len(present_days) + len(absent_days)
    pct = round((len(present_days) / total) * 100, 1) if total > 0 else 0

    col1, col2 = st.columns([1, 2])

    with col1:
        folder = os.path.join(DB_FOLDER, student_id)
        img_shown = False
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    st.image(os.path.join(folder, f), width=200)
                    img_shown = True
                    break
        if not img_shown:
            st.image("https://img.icons8.com/fluency/200/user-male-circle.png", width=200)

    with col2:
        st.markdown(f"### {student_id}")
        st.markdown(f"📧 **Email:** {info.get('email', '-')}")
        st.markdown(f"📱 **Phone:** {info.get('phone', '-')}")

        st.divider()

        m1, m2, m3 = st.columns(3)
        m1.metric("✅ Present Days", len(present_days))
        m2.metric("❌ Absent Days", len(absent_days))
        m3.metric("📊 Attendance %", f"{pct}%")

        if pct < 75:
            st.warning(f"⚠️ Attendance below 75% ({pct}%) — action required!")
        else:
            st.success(f"✅ Good attendance: {pct}%")


# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
