import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve
)

RANDOM_STATE = 42
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
# Convert target to 1 for malignant (positive class), 0 for benign.
y = (data.target == 0).astype(int)

print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print("Class counts (0 = benign, 1 = malignant):")
print(y.value_counts().sort_index())
X.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Model training complete.")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Benign (0)", "Malignant (1)"],
    yticklabels=["Benign (0)", "Malignant (1)"]
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
specificity = tn / (tn + fp) if (tn + fp) else 0.0
roc_auc = roc_auc_score(y_test, y_prob)

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall / Sensitivity", "Specificity", "F1-score", "ROC-AUC"],
    "Value": [accuracy, precision, recall, specificity, f1, roc_auc]
})
metrics.style.format({"Value": "{:.4f}"})

print(classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    zero_division=0,
    digits=4
))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

threshold_rows = []
for threshold in [0.30, 0.50, 0.70]:
    pred_at_threshold = (y_prob >= threshold).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, pred_at_threshold, labels=[0, 1]).ravel()
    threshold_rows.append({
        "Threshold": threshold,
        "Precision": precision_score(y_test, pred_at_threshold, zero_division=0),
        "Recall": recall_score(y_test, pred_at_threshold, zero_division=0),
        "Specificity": tn_t / (tn_t + fp_t) if (tn_t + fp_t) else 0.0,
        "F1-score": f1_score(y_test, pred_at_threshold, zero_division=0),
        "False positives": fp_t,
        "False negatives": fn_t
    })
threshold_results = pd.DataFrame(threshold_rows)
threshold_results.style.format({
    "Threshold": "{:.2f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
    "Specificity": "{:.3f}", "F1-score": "{:.3f}"
})
