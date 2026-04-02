import tkinter as tk
from tkinter import ttk, messagebox
import random
from datetime import datetime
from pathlib import Path
import string

APP_TITLE = "Smart Personal Assistant"

# Each activity runs as a standalone Tkinter app so students can click Run
# and immediately interact without command-line complexity.
class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("760x520")
        self.root.configure(bg="#f4f8ff")

        title_lbl = tk.Label(
            root,
            text=APP_TITLE,
            font=("Segoe UI", 20, "bold"),
            bg="#f4f8ff",
            fg="#1f3b73",
        )
        title_lbl.pack(pady=12)

        subtitle = tk.Label(
            root,
            text="Student Friendly Activity Interface",
            font=("Segoe UI", 11),
            bg="#f4f8ff",
            fg="#35528a",
        )
        subtitle.pack()

        self.main = tk.Frame(root, bg="#f4f8ff")
        self.main.pack(fill="both", expand=True, padx=20, pady=20)

        self.output = tk.Text(self.main, height=14, font=("Consolas", 10), bg="#ffffff")
        self.output.pack(side="bottom", fill="both", expand=True, pady=(12, 0))

        self.build_ui()

    def log(self, msg: str) -> None:
        self.output.insert("end", msg + "\n")
        self.output.see("end")

    def build_ui(self) -> None:
        idx = 1
        controls = tk.Frame(self.main, bg="#f4f8ff")
        controls.pack(fill="x")

        if idx == 1:
            tk.Button(controls, text="Greet Me", command=self.assistant_greet).pack(side="left", padx=6)
            tk.Button(controls, text="Show Time", command=self.assistant_time).pack(side="left", padx=6)
        elif idx == 2:
            tk.Label(controls, text="Choose: left/right", bg="#f4f8ff").pack(side="left")
            self.choice = ttk.Entry(controls, width=12)
            self.choice.pack(side="left", padx=6)
            tk.Button(controls, text="Play Story", command=self.story).pack(side="left", padx=6)
        elif idx == 3:
            tk.Label(controls, text="Password:", bg="#f4f8ff").pack(side="left")
            self.password = ttk.Entry(controls, show="*", width=20)
            self.password.pack(side="left", padx=6)
            tk.Button(controls, text="Validate", command=self.password_check).pack(side="left", padx=6)
        elif idx == 4:
            tk.Label(controls, text="Marks (comma separated):", bg="#f4f8ff").pack(side="left")
            self.marks = ttk.Entry(controls, width=24)
            self.marks.pack(side="left", padx=6)
            tk.Button(controls, text="Calculate Grade", command=self.grade_calc).pack(side="left", padx=6)
        elif idx == 5:
            tk.Label(controls, text="Amount:", bg="#f4f8ff").pack(side="left")
            self.amt = ttk.Entry(controls, width=12)
            self.amt.pack(side="left", padx=6)
            tk.Button(controls, text="Deposit", command=lambda: self.atm("deposit")).pack(side="left", padx=4)
            tk.Button(controls, text="Withdraw", command=lambda: self.atm("withdraw")).pack(side="left", padx=4)
            self.balance = 10000.0
        elif idx == 6:
            tk.Label(controls, text="Message:", bg="#f4f8ff").pack(side="left")
            self.message = ttk.Entry(controls, width=30)
            self.message.pack(side="left", padx=6)
            tk.Button(controls, text="Encode Morse", command=self.morse).pack(side="left", padx=6)
        elif idx == 7:
            tk.Label(controls, text="Item,Qty,Price", bg="#f4f8ff").pack(side="left")
            self.item = ttk.Entry(controls, width=24)
            self.item.pack(side="left", padx=6)
            tk.Button(controls, text="Add Invoice Line", command=self.invoice).pack(side="left", padx=6)
            self.invoice_total = 0.0
        elif idx == 8:
            tk.Button(controls, text="Generate Secret Number", command=self.new_secret).pack(side="left", padx=6)
            self.guess = ttk.Entry(controls, width=10)
            self.guess.pack(side="left", padx=6)
            tk.Button(controls, text="Guess", command=self.guess_game).pack(side="left", padx=6)
            self.secret = random.randint(1, 20)
        elif idx == 9:
            tk.Label(controls, text="Name:", bg="#f4f8ff").pack(side="left")
            self.student = ttk.Entry(controls, width=18)
            self.student.pack(side="left", padx=6)
            self.status = ttk.Combobox(controls, values=["Present", "Absent"], width=10, state="readonly")
            self.status.current(0)
            self.status.pack(side="left", padx=6)
            tk.Button(controls, text="Save Attendance", command=self.attendance).pack(side="left", padx=6)
        elif idx == 10:
            tk.Label(controls, text="KM:", bg="#f4f8ff").pack(side="left")
            self.km = ttk.Entry(controls, width=10)
            self.km.pack(side="left", padx=6)
            tk.Button(controls, text="Convert", command=self.convert).pack(side="left", padx=6)
        elif idx == 11:
            tk.Label(controls, text="Text:", bg="#f4f8ff").pack(side="left")
            self.cipher_text = ttk.Entry(controls, width=20)
            self.cipher_text.pack(side="left", padx=6)
            tk.Label(controls, text="Shift:", bg="#f4f8ff").pack(side="left")
            self.shift = ttk.Entry(controls, width=6)
            self.shift.pack(side="left", padx=6)
            tk.Button(controls, text="Encrypt/Decrypt", command=self.cipher).pack(side="left", padx=6)
        else:
            tk.Button(controls, text="Run Combined Demo", command=self.capstone).pack(side="left", padx=6)

        self.log("Click the buttons above to interact with this project.")

    def assistant_greet(self) -> None:
        self.log("Hello student! Keep practicing Python daily.")

    def assistant_time(self) -> None:
        self.log("Current time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def story(self) -> None:
        choice = self.choice.get().strip().lower()
        if choice == "left":
            self.log("You explored a cave and found treasure.")
        elif choice == "right":
            self.log("You followed the river and reached a village.")
        else:
            self.log("Choose either left or right to continue the story.")

    def password_check(self) -> None:
        pwd = self.password.get()
        issues = []
        if len(pwd) < 8:
            issues.append("min 8 chars")
        if not any(c.isupper() for c in pwd):
            issues.append("need uppercase")
        if not any(c.islower() for c in pwd):
            issues.append("need lowercase")
        if not any(c.isdigit() for c in pwd):
            issues.append("need digit")
        if not any(c in string.punctuation for c in pwd):
            issues.append("need symbol")
        if issues:
            self.log("Weak password: " + ", ".join(issues))
        else:
            self.log("Strong password!")

    def grade_calc(self) -> None:
        try:
            values = [float(x.strip()) for x in self.marks.get().split(",") if x.strip()]
            avg = sum(values) / len(values)
            grade = "A" if avg >= 90 else "B" if avg >= 80 else "C" if avg >= 70 else "D" if avg >= 60 else "F"
            self.log(f"Average: {avg:.2f}, Grade: {grade}")
        except Exception:
            self.log("Enter valid comma-separated marks, e.g. 85, 92, 78")

    def atm(self, mode: str) -> None:
        try:
            amount = float(self.amt.get())
            if amount <= 0:
                self.log("Amount must be positive.")
                return
            if mode == "deposit":
                self.balance += amount
                self.log(f"Deposited {amount:.2f}. Balance: {self.balance:.2f}")
            else:
                if amount > self.balance:
                    self.log("Insufficient balance.")
                else:
                    self.balance -= amount
                    self.log(f"Withdrew {amount:.2f}. Balance: {self.balance:.2f}")
        except ValueError:
            self.log("Enter a valid amount.")

    def morse(self) -> None:
        mapping = {"a": ".-", "b": "-...", "c": "-.-.", "d": "-..", "e": ".", "f": "..-.", "g": "--.", "h": "....", "i": "..", "j": ".---", "k": "-.-", "l": ".-..", "m": "--", "n": "-.", "o": "---", "p": ".--.", "q": "--.-", "r": ".-.", "s": "...", "t": "-", "u": "..-", "v": "...-", "w": ".--", "x": "-..-", "y": "-.--", "z": "--..", " ": "/"}
        msg = self.message.get().lower()
        encoded = " ".join(mapping.get(ch, "?") for ch in msg)
        self.log("Morse: " + encoded)

    def invoice(self) -> None:
        try:
            item, qty, price = [x.strip() for x in self.item.get().split(",")]
            line_total = float(qty) * float(price)
            self.invoice_total += line_total
            self.log(f"Added {item} -> {line_total:.2f} | Running total: {self.invoice_total:.2f}")
        except Exception:
            self.log("Format: Item,Qty,Price (example: Pen,2,15)")

    def new_secret(self) -> None:
        self.secret = random.randint(1, 20)
        self.log("Secret number reset between 1 and 20.")

    def guess_game(self) -> None:
        try:
            value = int(self.guess.get())
            if value == self.secret:
                self.log("Correct guess!")
            elif value < self.secret:
                self.log("Too low.")
            else:
                self.log("Too high.")
        except ValueError:
            self.log("Enter a valid integer guess.")

    def attendance(self) -> None:
        name = self.student.get().strip() or "Unknown"
        status = self.status.get()
        record = f"{datetime.now().isoformat()} | {name} | {status}\n"
        file_path = Path("attendance_log.txt")
        with file_path.open("a", encoding="utf-8") as f:
            f.write(record)
        self.log("Attendance saved to attendance_log.txt")

    def convert(self) -> None:
        try:
            km = float(self.km.get())
            self.log(f"{km} km = {km * 0.621371:.3f} miles")
        except ValueError:
            self.log("Enter a valid numeric KM value.")

    def cipher(self) -> None:
        text = self.cipher_text.get()
        try:
            shift = int(self.shift.get())
        except ValueError:
            self.log("Shift must be an integer.")
            return
        alphabet = string.ascii_lowercase
        result = []
        for ch in text:
            if ch.lower() in alphabet:
                i = alphabet.index(ch.lower())
                new_ch = alphabet[(i + shift) % 26]
                result.append(new_ch.upper() if ch.isupper() else new_ch)
            else:
                result.append(ch)
        encrypted = "".join(result)
        self.log("Encrypted: " + encrypted)

    def capstone(self) -> None:
        self.log("Capstone demo:")
        self.log("- password check sample")
        self.log("- converter sample")
        self.log("- guess game sample")
        self.log("Open individual activities to practice each concept deeply.")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
