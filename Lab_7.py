# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)

RANDOM_STATE = 42
sns.set_theme(style='whitegrid')
print('Imports complete.')
X, y = make_classification(
    n_samples=1200, n_features=12, n_informative=6, n_redundant=2,
    n_classes=2, class_sep=1.0, flip_y=0.04, random_state=RANDOM_STATE
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
print('Training samples:', len(y_train), '| Test samples:', len(y_test))
print('Class counts in full dataset:', np.bincount(y))

# Compare a single decision tree with Random Forest models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest (10 trees)': RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1),
    'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'Random Forest (300 trees)': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append({
        'Model': name,
        'Train accuracy': model.score(X_train, y_train),
        'Test accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1-score': f1_score(y_test, pred, zero_division=0),
    })
results_df = pd.DataFrame(results)
display(results_df.round(3))

rf_classifier = models['Random Forest (100 trees)']
y_pred = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted class'); plt.ylabel('True class')
plt.title('Random Forest confusion matrix')
plt.show()

importance = pd.Series(rf_classifier.feature_importances_, index=[f'Feature {i+1}' for i in range(X.shape[1])])
importance = importance.sort_values(ascending=False)
display(importance.rename('Importance').to_frame().round(4))
plt.figure(figsize=(8, 4))
importance.sort_values().plot(kind='barh', color='teal')
plt.xlabel('Impurity-based importance'); plt.title('Random Forest feature importance')
plt.tight_layout(); plt.show()

cv_models = {
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
}
cv_rows = []
for name, model in cv_models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_rows.append({'Model': name, 'Fold accuracies': np.round(scores, 3),
                    'Mean CV accuracy': scores.mean(), 'Std. deviation': scores.std()})
for row in cv_rows:
    print(row['Model'], '| folds:', row['Fold accuracies'],
          '| mean:', round(row['Mean CV accuracy'], 3),
          '| std:', round(row['Std. deviation'], 3))

X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10, n_informative=6, noise=18, random_state=RANDOM_STATE
)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=RANDOM_STATE
)
reg_models = {
    'Shallow forest (max_depth=4)': RandomForestRegressor(n_estimators=150, max_depth=4, random_state=RANDOM_STATE, n_jobs=-1),
    'Unrestricted forest': RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
}
reg_rows = []
for name, model in reg_models.items():
    model.fit(Xr_train, yr_train)
    pred = model.predict(Xr_test)
    reg_rows.append({'Model': name, 'MAE': mean_absolute_error(yr_test, pred),
                     'RMSE': np.sqrt(mean_squared_error(yr_test, pred)),
                     'R²': r2_score(yr_test, pred)})
display(pd.DataFrame(reg_rows).round(3))
