from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 1. Load and split dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training examples:", X_train.shape[0])
print("Test examples:", X_test.shape[0])
print("Features:", iris.feature_names)
print("Classes:", list(iris.target_names))
print("-" * 40)

# 2. Define pipeline with preprocessing and a placeholder classifier
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 3. Define hyperparameter grid for multiple models
param_grid = [
    {
        "classifier": [LogisticRegression(max_iter=1000)],
        "classifier__C": [0.1, 1, 10]
    },
    {
        "classifier": [SVC()],
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["linear", "rbf"]
    },
    {
        "classifier": [RandomForestClassifier(random_state=42)],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 3, 5]
    }
]

# 4. Set up Grid Search with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

# 5. Fit model
search.fit(X_train, y_train)

print("Best classifier:", search.best_estimator_.named_steps["classifier"])
print("Best parameters:", search.best_params_)
print(f"Best mean cross-validation accuracy: {search.best_score_:.3f}")
print("-" * 40)

# 6. Display top cross-validation results (Wrapped in print() so it shows in terminal scripts)
results = pd.DataFrame(search.cv_results_)
top_results = results[["mean_test_score", "std_test_score", "mean_train_score", "params"]].sort_values(
    by="mean_test_score", ascending=False
).head(10).reset_index(drop=True)

print("Top CV Results:")
print(top_results)
print("-" * 40)

# 7. Evaluate on the test set
y_pred = search.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
