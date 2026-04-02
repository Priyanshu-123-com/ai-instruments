import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import os

APP_TITLE = "Language Translator Bot"


# Student-friendly UI for computer vision/NLP projects.
# Clicking "Start Demo" opens webcam processing and "q" quits camera window.
class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("780x460")
        self.root.configure(bg="#fff5ee")

        tk.Label(root, text=APP_TITLE, font=("Segoe UI", 20, "bold"), bg="#fff5ee", fg="#8a3b12").pack(pady=12)
        tk.Label(root, text="AI Architect Activity", font=("Segoe UI", 11), bg="#fff5ee", fg="#9a4f2b").pack()

        tk.Button(root, text="Start Demo", command=self.start_demo).pack(pady=10)
        tk.Button(root, text="Show Setup Tips", command=self.show_tips).pack()

        self.output = tk.Text(root, height=12, font=("Consolas", 10))
        self.output.pack(fill="both", expand=True, padx=16, pady=12)
        self.log("Press 'Start Demo' to launch a simple CV processing loop.")

    def log(self, msg: str) -> None:
        self.output.insert("end", msg + "\n")
        self.output.see("end")

    def show_tips(self) -> None:
        tips = (
            "1) Install dependencies from requirements.txt\n"
            "2) Enable webcam access in OS privacy settings\n"
            "3) Close apps already using camera\n"
            "4) Press q in OpenCV window to exit"
        )
        messagebox.showinfo("Setup Tips", tips)

    def start_demo(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log("Webcam not accessible. Check permissions.")
            return
        self.log("Webcam opened. Press q to quit demo window.")
        while True:
            ok, frame = cap.read()
            if not ok:
                self.log("Frame capture failed.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 140)
            cv2.imshow(APP_TITLE + " - Original", frame)
            cv2.imshow(APP_TITLE + " - Processed", edges)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.log("Demo ended.")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
