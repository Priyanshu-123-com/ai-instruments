import csv
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2

APP_TITLE = "Camera Attendance System"
ATTENDANCE_FILE = Path("attendance.csv")
SNAPSHOT_DIR = Path("snapshots")


class AttendanceApp:
    # Purpose-built, student-friendly app for camera-based attendance.
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("860x560")
        self.root.configure(bg="#f0f7ff")

        tk.Label(
            root,
            text=APP_TITLE,
            font=("Segoe UI", 20, "bold"),
            bg="#f0f7ff",
            fg="#0f3d7a",
        ).pack(pady=10)

        top = tk.Frame(root, bg="#f0f7ff")
        top.pack(fill="x", padx=18, pady=8)

        tk.Label(top, text="Student Name:", bg="#f0f7ff", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.name_entry = ttk.Entry(top, width=24)
        self.name_entry.pack(side="left", padx=8)

        tk.Button(top, text="Capture & Mark Present", command=self.capture_and_mark, bg="#2d7ff9", fg="white").pack(side="left", padx=6)
        tk.Button(top, text="Show Today's Summary", command=self.show_summary).pack(side="left", padx=6)

        self.log_box = tk.Text(root, font=("Consolas", 10), bg="white", height=20)
        self.log_box.pack(fill="both", expand=True, padx=18, pady=12)

        self.ensure_attendance_file()
        self.log("Ready: enter a student name, then click 'Capture & Mark Present'.")

    def log(self, message: str) -> None:
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")

    def ensure_attendance_file(self) -> None:
        if not ATTENDANCE_FILE.exists():
            with ATTENDANCE_FILE.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "name", "status", "snapshot_path"])

    def capture_face_snapshot(self, student_name: str) -> str | None:
        # Haar cascade is built into OpenCV package and works offline.
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            self.log("Error: Camera could not be opened.")
            return None

        snapshot_path = None
        self.log("Camera opened. Press SPACE to capture, or ESC to cancel.")

        while True:
            ok, frame = camera.read()
            if not ok:
                self.log("Error: Could not read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 205, 50), 2)

            cv2.putText(frame, "SPACE: capture | ESC: cancel", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(APP_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                self.log("Capture cancelled.")
                break
            if key == 32:
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"{student_name.replace(' ', '_')}_{timestamp}.jpg"
                snapshot_path = str(SNAPSHOT_DIR / file_name)
                cv2.imwrite(snapshot_path, frame)
                self.log(f"Snapshot saved: {snapshot_path}")
                break

        camera.release()
        cv2.destroyAllWindows()
        return snapshot_path

    def capture_and_mark(self) -> None:
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Missing Name", "Please enter a student name first.")
            return

        snapshot = self.capture_face_snapshot(name)
        if snapshot is None:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with ATTENDANCE_FILE.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([now, name, "Present", snapshot])

        self.log(f"Attendance marked for {name} at {now}.")
        messagebox.showinfo("Success", f"Attendance saved for {name}.")

    def show_summary(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        count = 0
        names: list[str] = []
        with ATTENDANCE_FILE.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["timestamp"].startswith(today):
                    count += 1
                    names.append(row["name"])

        unique_names = sorted(set(names))
        self.log(f"Today's attendance count: {count}")
        self.log("Students: " + (", ".join(unique_names) if unique_names else "No records yet"))


def main() -> None:
    root = tk.Tk()
    AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
