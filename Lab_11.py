import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

# 1. Load Dataset
cancer = load_breast_cancer(as_frame=True)
X = cancer.data
y = cancer.target

print("Feature matrix:", X.shape)
print("Class names:", list(cancer.target_names))
print("Class counts:")
print(y.value_counts().rename(index=dict(enumerate(cancer.target_names))))
display(X.head())

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# 3. Baseline SVC Model
baseline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)


def metric_row(name, y_true, y_pred):
  return {
      "Model": name,
      "Accuracy": accuracy_score(y_true, y_pred),
      "Precision": precision_score(y_true, y_pred),
      "Recall": recall_score(y_true, y_pred),
      "F1": f1_score(y_true, y_pred),
  }


results = [metric_row("Baseline SVC", y_test, baseline_pred)]
display(pd.DataFrame(results).set_index("Model").round(4))

# 4. Hyperparameter Tuning with GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid_params = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1],
    "svc__kernel": ["rbf", "linear"],
}

grid_search = GridSearchCV(
    estimator=Pipeline([("scaler", StandardScaler()), ("svc", SVC())]),
    param_grid=grid_params,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)
grid_search.fit(X_train, y_train)
print("Best CV F1 (GridSearchCV):", round(grid_search.best_score_, 4))
print("Best parameters (GridSearchCV):", grid_search.best_params_)

# 5. Hyperparameter Tuning with RandomizedSearchCV
random_params = {
    "svc__C": loguniform(1e-2, 1e3),
    "svc__gamma": loguniform(1e-5, 1e0),
    "svc__kernel": ["rbf", "linear"],
}

random_search = RandomizedSearchCV(
    estimator=Pipeline([("scaler", StandardScaler()), ("svc", SVC())]),
    param_distributions=random_params,
    n_iter=30,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    refit=True,
    return_train_score=True,
)
random_search.fit(X_train, y_train)
print("Best CV F1 (RandomizedSearchCV):", round(random_search.best_score_, 4))
print("Best parameters (RandomizedSearchCV):", random_search.best_params_)

# 6. Model Evaluation and Comparison
grid_pred = grid_search.predict(X_test)
random_pred = random_search.predict(X_test)

results = [
    metric_row("Baseline SVC", y_test, baseline_pred),
    metric_row("GridSearchCV SVC", y_test, grid_pred),
    metric_row("RandomizedSearchCV SVC", y_test, random_pred),
]

comparison = pd.DataFrame(results).set_index("Model").round(4)
display(comparison)

# 7. Visualization
ax = comparison[["Accuracy", "Precision", "Recall", "F1"]].plot(
    kind="bar", figsize=(10, 5), ylim=(0.85, 1.0), rot=0
)
ax.set_ylabel("Test-set score")
ax.set_title("Baseline and tuned SVC performance")
ax.grid(axis="y", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 8. Detailed Report for Best Model
best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test)
print("\nClassification report (GridSearchCV model):")
print(classification_report(y_test, best_pred, target_names=cancer.target_names))
print("Confusion matrix:")
print(confusion_matrix(y_test, best_pred))

# 9. Top Cross-Validation Results Inspection
cv_results = pd.DataFrame(grid_search.cv_results_)
cols = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "param_svc__C",
    "param_svc__gamma",
    "param_svc__kernel",
]
display(
    cv_results[cols]
    .sort_values("rank_test_score")
    .head(10)
    .reset_index(drop=True)
    .round(4)
)