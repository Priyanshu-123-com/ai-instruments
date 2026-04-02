import tkinter as tk
from tkinter import ttk

from textblob import TextBlob

APP_TITLE = "Sentiment Analyzer (Text)"


class SentimentApp:
    # Text-focused UI: no webcam, no heavy interface noise.
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("800x520")
        self.root.configure(bg="#f9f7ff")

        tk.Label(
            root,
            text=APP_TITLE,
            font=("Segoe UI", 20, "bold"),
            bg="#f9f7ff",
            fg="#4b2a88",
        ).pack(pady=10)

        tk.Label(
            root,
            text="Enter a sentence and click Analyze",
            font=("Segoe UI", 11),
            bg="#f9f7ff",
            fg="#5d45a0",
        ).pack()

        frame = tk.Frame(root, bg="#f9f7ff")
        frame.pack(fill="x", padx=20, pady=12)

        self.input_text = tk.Text(frame, height=8, font=("Segoe UI", 11), wrap="word")
        self.input_text.pack(fill="x")

        actions = tk.Frame(root, bg="#f9f7ff")
        actions.pack(fill="x", padx=20)
        ttk.Button(actions, text="Analyze", command=self.analyze).pack(side="left")
        ttk.Button(actions, text="Clear", command=self.clear).pack(side="left", padx=8)

        self.output = tk.Text(root, height=12, font=("Consolas", 10), bg="white")
        self.output.pack(fill="both", expand=True, padx=20, pady=14)

    def log(self, text: str) -> None:
        self.output.insert("end", text + "\n")
        self.output.see("end")

    def clear(self) -> None:
        self.input_text.delete("1.0", "end")
        self.output.delete("1.0", "end")

    def analyze(self) -> None:
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            self.log("Please enter text to analyze.")
            return

        analysis = TextBlob(text).sentiment
        polarity = analysis.polarity
        subjectivity = analysis.subjectivity

        if polarity > 0.15:
            label = "Positive"
        elif polarity < -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        self.log("Input: " + text)
        self.log(f"Polarity: {polarity:.3f}")
        self.log(f"Subjectivity: {subjectivity:.3f}")
        self.log("Sentiment Label: " + label)
        self.log("-" * 42)


def main() -> None:
    root = tk.Tk()
    SentimentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
