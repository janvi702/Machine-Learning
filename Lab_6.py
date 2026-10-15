# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
print("Imports complete.")
X_binary, y_binary = make_moons(n_samples=600, noise=0.25, random_state=RANDOM_STATE)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_binary, y_binary, test_size=0.25, random_state=RANDOM_STATE, stratify=y_binary
)

plt.figure(figsize=(6, 5))
plt.scatter(X_binary[:, 0], X_binary[:, 1], c=y_binary, cmap="coolwarm", s=22, edgecolor="k", linewidth=0.25)
plt.title("Binary dataset: two moons")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.show()
print("Train size:", len(yb_train), "| Test size:", len(yb_test))
binary_models = {
    "Linear": SVC(kernel="linear", C=1.0),
    "Polynomial (degree 3)": SVC(kernel="poly", degree=3, C=1.0, gamma="scale"),
    "RBF": SVC(kernel="rbf", C=1.0, gamma="scale"),
}

binary_rows = []
for name, estimator in binary_models.items():
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(Xb_train, yb_train)
    pred = model.predict(Xb_test)
    binary_rows.append({
        "Kernel": name,
        "Accuracy": accuracy_score(yb_test, pred),
        "Precision (macro)": precision_score(yb_test, pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(yb_test, pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(yb_test, pred, average="macro", zero_division=0),
        "model": model,
        "pred": pred,
    })

binary_results = pd.DataFrame([{k:v for k,v in row.items() if k not in ("model", "pred")} for row in binary_rows])
display(binary_results.sort_values("F1 (macro)", ascending=False).round(3))
# Visualize each model's decision regions. The displayed axes use standardized coordinates.
def plot_decision_regions(ax, model, X, y, title):
    scaler = model.named_steps["standardscaler"]
    svc = model.named_steps["svc"]
    X_scaled = scaler.transform(X)
    pad = 0.6
    x0_min, x0_max = X_scaled[:, 0].min() - pad, X_scaled[:, 0].max() + pad
    x1_min, x1_max = X_scaled[:, 1].min() - pad, X_scaled[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 250), np.linspace(x1_min, x1_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = svc.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.22, cmap="coolwarm")
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="coolwarm", s=18, edgecolor="k", linewidth=0.2)
    # Mark support vectors in standardized coordinates
    sv = svc.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=75, facecolors="none", edgecolors="black", linewidths=1.0, label="Support vectors")
    ax.set_title(title); ax.set_xlabel("Standardized feature 1"); ax.set_ylabel("Standardized feature 2")
    ax.legend(loc="best", fontsize=8)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, row in zip(axes, binary_rows):
    plot_decision_regions(ax, row["model"], Xb_train, yb_train, row["Kernel"])
plt.tight_layout(); plt.show()
X_multi, y_multi = make_classification(
    n_samples=900, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, n_classes=3, class_sep=1.35,
    random_state=RANDOM_STATE
)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_multi, y_multi, test_size=0.25, random_state=RANDOM_STATE, stratify=y_multi
)

multi_model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
multi_model.fit(Xm_train, ym_train)
ym_pred = multi_model.predict(Xm_test)

print("Multiclass classification report")
print(classification_report(ym_test, ym_pred, digits=3, zero_division=0))
cm = confusion_matrix(ym_test, ym_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=np.unique(y_multi), yticklabels=np.unique(y_multi))
plt.xlabel("Predicted class"); plt.ylabel("True class"); plt.title("Multiclass RBF SVM confusion matrix")
plt.show()
tuning_rows = []
for C in [0.1, 1, 10]:
    for gamma in [0.1, 1, 10]:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma))
        model.fit(Xb_train, yb_train)
        pred = model.predict(Xb_test)
        tuning_rows.append({"C": C, "gamma": gamma, "Accuracy": accuracy_score(yb_test, pred),
                           "F1 (macro)": f1_score(yb_test, pred, average="macro")})
tuning_results = pd.DataFrame(tuning_rows).sort_values("F1 (macro)", ascending=False)
display(tuning_results.round(3))
