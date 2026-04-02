import tkinter as tk
from tkinter import ttk
import json
from pathlib import Path
import re

APP_TITLE = "Personal Study Buddy AI"

# Lightweight offline model strategy:
# Retrieval-based chatbot using TF-IDF cosine similarity.
# It is open-source, fast, and works without API keys.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_FILE = Path("knowledge_base.json")


class StudyBuddyApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("860x560")
        self.root.configure(bg="#fffef5")

        tk.Label(root, text=APP_TITLE, font=("Segoe UI", 20, "bold"), bg="#fffef5", fg="#7d5610").pack(pady=10)
        tk.Label(root, text="Offline Lightweight AI Tutor", font=("Segoe UI", 11), bg="#fffef5", fg="#8f6a23").pack()

        top = tk.Frame(root, bg="#fffef5")
        top.pack(fill="x", padx=18, pady=10)

        self.question_entry = ttk.Entry(top, width=80)
        self.question_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(top, text="Ask", command=self.answer_question).pack(side="left", padx=8)

        ttk.Button(top, text="Load/Reset KB", command=self.load_knowledge_base).pack(side="left")

        self.output = tk.Text(root, height=22, font=("Consolas", 10), bg="white")
        self.output.pack(fill="both", expand=True, padx=18, pady=12)

        self.questions: list[str] = []
        self.answers: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.question_matrix = None
        self.load_knowledge_base()

    def log(self, msg: str) -> None:
        self.output.insert("end", msg + "\n")
        self.output.see("end")

    def build_default_knowledge(self) -> None:
        default_data = {
            "faqs": [
                {"q": "What is Python?", "a": "Python is a beginner-friendly programming language used for automation, web, AI, and data science."},
                {"q": "What is a variable?", "a": "A variable stores data in memory so we can reuse and change values in a program."},
                {"q": "What is machine learning?", "a": "Machine learning is a way for computers to learn patterns from data and make predictions."},
                {"q": "Difference between list and tuple?", "a": "Lists are mutable (changeable), tuples are immutable (fixed after creation)."},
                {"q": "How do loops work?", "a": "Loops repeat a block of code until a condition ends or all items are processed."},
            ]
        }
        FAQ_FILE.write_text(json.dumps(default_data, indent=2), encoding="utf-8")

    def load_knowledge_base(self) -> None:
        if not FAQ_FILE.exists():
            self.build_default_knowledge()
        data = json.loads(FAQ_FILE.read_text(encoding="utf-8"))
        faqs = data.get("faqs", [])
        self.questions = [item["q"] for item in faqs if "q" in item and "a" in item]
        self.answers = [item["a"] for item in faqs if "q" in item and "a" in item]
        if not self.questions:
            self.log("Knowledge base is empty. Add FAQ items in knowledge_base.json.")
            return
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.question_matrix = self.vectorizer.fit_transform(self.questions)
        self.log("Knowledge base loaded. Ask a question.")

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text)

    def answer_question(self) -> None:
        user_q = self.normalize(self.question_entry.get())
        if not user_q:
            self.log("Please type a question.")
            return
        if self.vectorizer is None or self.question_matrix is None:
            self.log("Knowledge model not ready. Reload KB.")
            return

        q_vec = self.vectorizer.transform([user_q])
        sims = cosine_similarity(q_vec, self.question_matrix)[0]
        best_index = int(sims.argmax())
        best_score = float(sims[best_index])

        self.log("You: " + self.question_entry.get().strip())
        if best_score < 0.15:
            self.log("Study Buddy: I am not sure yet. Try rephrasing or update knowledge_base.json.")
        else:
            self.log("Study Buddy: " + self.answers[best_index])
            self.log(f"(match confidence: {best_score:.2f})")
        self.log("-" * 50)


def main() -> None:
    root = tk.Tk()
    StudyBuddyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
