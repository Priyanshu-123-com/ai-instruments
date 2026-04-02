import tkinter as tk
from tkinter import ttk
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

APP_TITLE = "Cricket Stats Analyzer"

# Student-friendly analytics app:
# - Generates local data automatically
# - Lets student click "Run Analysis" and observe outputs/plots
def generate_data(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "feature_a": rng.normal(50, 10, 120),
        "feature_b": rng.normal(30, 8, 120),
        "target": rng.normal(70, 12, 120),
    })
    df.to_csv(path, index=False)
    return df


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("860x560")
        self.root.configure(bg="#eef6ff")

        tk.Label(root, text=APP_TITLE, font=("Segoe UI", 20, "bold"), bg="#eef6ff", fg="#1b4b8a").pack(pady=12)
        tk.Label(root, text="Data Science Learning Activity", font=("Segoe UI", 11), bg="#eef6ff", fg="#2b5f9a").pack()

        top = tk.Frame(root, bg="#eef6ff")
        top.pack(fill="x", padx=16, pady=10)
        tk.Button(top, text="Run Analysis", command=self.run_analysis).pack(side="left")
        tk.Button(top, text="Plot Graph", command=self.plot).pack(side="left", padx=8)

        self.text = tk.Text(root, font=("Consolas", 10), bg="white")
        self.text.pack(fill="both", expand=True, padx=16, pady=12)

        self.data_file = Path("dummy_data.csv")
        self.df = None
        self.log("Click 'Run Analysis' to generate/load data and inspect statistics.")

    def log(self, msg: str) -> None:
        self.text.insert("end", msg + "\n")
        self.text.see("end")

    def run_analysis(self) -> None:
        if self.data_file.exists():
            self.df = pd.read_csv(self.data_file)
            self.log("Loaded existing dataset: dummy_data.csv")
        else:
            self.df = generate_data(self.data_file)
            self.log("Generated dataset: dummy_data.csv")
        self.df["combined_score"] = self.df["feature_a"] * 0.6 + self.df["feature_b"] * 0.4
        self.log("\nFirst 5 rows:")
        self.log(self.df.head().to_string(index=False))
        self.log("\nSummary:")
        self.log(self.df.describe().to_string())

    def plot(self) -> None:
        if self.df is None:
            self.run_analysis()
        plt.figure(figsize=(8, 5))
        plt.scatter(self.df["combined_score"], self.df["target"], alpha=0.7)
        plt.title(APP_TITLE + " - Learning Plot")
        plt.xlabel("Combined Score")
        plt.ylabel("Target")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        self.log("Plot displayed successfully.")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
