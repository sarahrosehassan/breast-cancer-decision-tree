"""Train a decision tree to tell benign from malignant breast tumors.

Every idea used here is explained with pictures in README.md.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42  # fixed so every run gives the same split and the same tree


def accuracy(model: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray) -> float:
    return accuracy_score(y, model.predict(X))


def train_basic_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    # Groups smaller than 6 become leaves, so the tree can't chase single odd examples
    model = DecisionTreeClassifier(criterion="entropy", min_samples_split=6, random_state=SEED)
    return model.fit(X_train, y_train)


def plot_depth_vs_accuracy(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> None:
    depths = range(1, 16)
    train_scores = []
    test_scores = []
    for depth in depths:
        model = DecisionTreeClassifier(
            criterion="entropy", max_depth=depth, min_samples_split=6, random_state=SEED
        )
        model.fit(X_train, y_train)
        train_scores.append(accuracy(model, X_train, y_train))
        test_scores.append(accuracy(model, X_test, y_test))

    plt.figure()
    plt.plot(depths, train_scores, "o-", label="Training accuracy")
    plt.plot(depths, test_scores, "s--", label="Test accuracy")
    plt.xlabel("Maximum tree depth")
    plt.ylabel("Accuracy")
    plt.title("Deeper trees memorize the training data")
    plt.legend()


def tune_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    # Settings are chosen with 10-fold cross-validation on training data only,
    # so the test set stays unseen and its score stays honest
    search = GridSearchCV(
        DecisionTreeClassifier(criterion="entropy", random_state=SEED),
        param_grid={"max_depth": range(1, 16), "min_samples_split": range(2, 21)},
        cv=10,
    )
    search.fit(X_train, y_train)
    best = search.best_params_
    print(f"Best settings: max_depth={best['max_depth']}, min_samples_split={best['min_samples_split']}")
    return search.best_estimator_


def show_tree(model: DecisionTreeClassifier, feature_names: list[str], title: str) -> None:
    plt.figure(figsize=(14, 8))
    plot_tree(model, feature_names=feature_names, class_names=["malignant", "benign"], filled=True, fontsize=8)
    plt.title(title)


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    print(f"{X.shape[0]} tumor samples, {X.shape[1]} measurements each")

    # stratify=y keeps the benign/malignant mix the same in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED
    )

    basic = train_basic_tree(X_train, y_train)
    print(f"Basic tree test accuracy: {accuracy(basic, X_test, y_test):.1%}")
    show_tree(basic, feature_names, "Basic tree")

    plot_depth_vs_accuracy(X_train, X_test, y_train, y_test)

    tuned = tune_tree(X_train, y_train)
    print(f"Tuned tree test accuracy: {accuracy(tuned, X_test, y_test):.1%}")
    show_tree(tuned, feature_names, "Tuned tree")

    plt.show()


if __name__ == "__main__":
    main()
