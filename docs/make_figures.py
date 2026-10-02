"""Generate the figures used in the README.

Run from the repo root:
    .venv/bin/python docs/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

OUT_DIR = Path(__file__).parent / "images"

# Colors (validated categorical slots 1 and 2, plus neutral text/surface tokens)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#d9d8d3"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED,
    "axes.titlecolor": TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "font.size": 11,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT_DIR / name)


def split_figure(y: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Horizontal bars showing a stratified 60/40 split keeps the class mix."""
    rows = [("All data", y), ("Training (60%)", y_train), ("Test (40%)", y_test)]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for i, (label, labels) in enumerate(reversed(rows)):
        benign = np.mean(labels == 1)
        ax.barh(i, benign, color=BLUE, height=0.55)
        ax.barh(i, 1 - benign, left=benign + 0.004, color=ORANGE, height=0.55)
        ax.text(benign / 2, i, f"benign {benign:.0%}", ha="center", va="center", color="white", fontsize=10)
        ax.text(benign + (1 - benign) / 2, i, f"malignant {1 - benign:.0%}", ha="center", va="center", color="white", fontsize=10)
        ax.text(-0.02, i, f"{label}\n{len(labels)} rows", ha="right", va="center", color=TEXT, fontsize=10)
    ax.set_xlim(0, 1.005)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    ax.set_title("Stratified split: every piece keeps the same class mix", loc="left")
    save(fig, "stratified_split.png")


def entropy_figure() -> None:
    """Entropy of a two-class group as the share of one class changes."""
    p = np.linspace(0.001, 0.999, 400)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(p, h, color=BLUE, linewidth=2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.scatter([0, 0.5, 1], [0, 1, 0], s=50, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.text(0.5, 1.07, "50/50 mix\nentropy 1 (messiest)", ha="center", color=TEXT_MUTED, fontsize=9)
    ax.text(0.07, 0.05, "all one class\nentropy 0 (pure)", ha="left", color=TEXT_MUTED, fontsize=9)
    ax.text(0.93, 0.05, "all the other class\nentropy 0 (pure)", ha="right", color=TEXT_MUTED, fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Share of the group that is class A")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Entropy measures how mixed a group is", loc="left")
    save(fig, "entropy.png")


def depth_figure(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Training vs test accuracy as the tree is allowed to grow deeper."""
    depths = list(range(1, 16))
    train_acc: list[float] = []
    test_acc: list[float] = []
    for depth in depths:
        model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth, min_samples_split=6, random_state=42)
        model.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, model.predict(X_train)))
        test_acc.append(accuracy_score(y_test, model.predict(X_test)))

    best = depths[int(np.argmax(test_acc))]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvspan(0.5, best - 0.5, color=NEUTRAL, alpha=0.35, linewidth=0)
    ax.axvspan(best + 0.5, 15.5, color=NEUTRAL, alpha=0.2, linewidth=0)
    ax.text(1, 0.878, "underfitting", color=TEXT_MUTED, fontsize=9)
    ax.text(9.5, 0.878, "overfitting: memorizing, not learning", color=TEXT_MUTED, fontsize=9)
    ax.plot(depths, train_acc, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=2, label="Training accuracy")
    ax.plot(depths, test_acc, color=ORANGE, linewidth=2, marker="s", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=2, label="Test accuracy (unseen data)")
    ax.annotate(f"best on unseen data\n(depth {best})", (best, max(test_acc)), xytext=(best + 2.3, 0.948),
                textcoords="data", color=TEXT, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "linewidth": 1})
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(0.87, 1.01)
    ax.set_xticks(depths)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Maximum tree depth (how many questions deep)")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.set_title("Deeper trees ace the training data but stop improving on new data", loc="left")
    save(fig, "depth_vs_accuracy.png")


def tree_figure(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> None:
    """A small, readable tree (depth 2) so the flowchart idea is visible."""
    model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(11, 5))
    tree.plot_tree(model, feature_names=feature_names, class_names=["malignant", "benign"], filled=True,
                   rounded=True, impurity=True, fontsize=10, ax=ax)
    ax.set_title("A decision tree is a flowchart of yes/no questions (left = yes, right = no)", loc="left")
    save(fig, "decision_tree.png")


def cross_validation_figure(k: int = 10) -> None:
    """Grid of k rounds; each round holds out a different fold for validation."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for round_idx in range(k):
        for fold in range(k):
            is_val = fold == round_idx
            ax.add_patch(Rectangle((fold + 0.06, k - 1 - round_idx + 0.1), 0.88, 0.8,
                                   color=ORANGE if is_val else BLUE, alpha=1 if is_val else 0.55, linewidth=0))
        ax.text(-0.2, k - 1 - round_idx + 0.5, f"round {round_idx + 1}", ha="right", va="center",
                color=TEXT_MUTED, fontsize=9)
    ax.text(0.06, k + 0.35, "■ train on these 9 folds", color=BLUE, fontsize=10)
    ax.text(4.2, k + 0.35, "■ check on this 1 fold", color=ORANGE, fontsize=10)
    ax.text(k / 2, -0.6, "final score = average of the 10 checks", ha="center", color=TEXT, fontsize=10)
    ax.set_xlim(-1.4, k)
    ax.set_ylim(-1, k + 1)
    ax.axis("off")
    ax.set_title("10-fold cross-validation: every fold gets a turn as the quiz", loc="left")
    save(fig, "cross_validation.png")


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

    split_figure(y, y_train, y_test)
    entropy_figure()
    depth_figure(X_train, X_test, y_train, y_test)
    tree_figure(X_train, y_train, list(data.feature_names))
    cross_validation_figure()


if __name__ == "__main__":
    main()
