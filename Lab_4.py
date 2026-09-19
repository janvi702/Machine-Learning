import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve
)

sns.set_theme(style='whitegrid')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
def sigmoid(z):
    # Clipping avoids overflow when z is very large in magnitude.
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-8, 8, 400)
plt.figure(figsize=(8, 4))
plt.plot(z_values, sigmoid(z_values), linewidth=2.5)
plt.axhline(0.5, color='gray', linestyle='--', label='0.5 threshold')
plt.axvline(0, color='gray', linestyle=':')
plt.xlabel('Linear score z')
plt.ylabel('P(y = 1 | x)')
plt.title('Sigmoid function')
plt.legend()
plt.show()
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
# sklearn encodes malignant as 0 and benign as 1; invert it so 1 = malignant.
y = (data.target == 0).astype(int)

print(f'Samples: {X.shape[0]}')
print(f'Features: {X.shape[1]}')
print('Class counts:')
print(pd.Series(y, name='malignant').value_counts().rename({0: 'benign', 1: 'malignant'}))
display(X.head())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Training shape:', X_train_scaled.shape)
print('Testing shape: ', X_test_scaled.shape)
print('Training malignant rate:', round(y_train.mean(), 3))
print('Testing malignant rate: ', round(y_test.mean(), 3))

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.05, epochs=2500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.loss_history = []

    def _add_intercept(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        Xb = self._add_intercept(X)
        y = np.asarray(y).reshape(-1)
        self.weights = np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            probabilities = sigmoid(Xb @ self.weights)
            probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
            loss = -np.mean(y * np.log(probabilities) + (1-y) * np.log(1-probabilities))
            gradient = (Xb.T @ (probabilities - y)) / len(y)
            self.weights -= self.learning_rate * gradient
            self.loss_history.append(loss)
        return self

    def predict_proba(self, X):
        Xb = self._add_intercept(X)
        p1 = sigmoid(Xb @ self.weights)
        return np.c_[1-p1, p1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

scratch_model = LogisticRegressionScratch(learning_rate=0.05, epochs=2500)
scratch_model.fit(X_train_scaled, y_train)

plt.figure(figsize=(8, 4))
plt.plot(scratch_model.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Binary cross-entropy loss')
plt.title('Training loss from scratch')
plt.show()

def show_metrics(y_true, y_pred, y_score, title='Model evaluation'):
    results = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_score),
    }
    display(pd.DataFrame(results, index=[title]).round(4))
    print(classification_report(y_true, y_pred, target_names=['benign (0)', 'malignant (1)'], zero_division=0))
    return results

scratch_scores = scratch_model.predict_proba(X_test_scaled)[:, 1]
scratch_predictions = scratch_model.predict(X_test_scaled)
scratch_metrics = show_metrics(y_test, scratch_predictions, scratch_scores, 'From-scratch model')
cm = confusion_matrix(y_test, scratch_predictions)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted benign', 'Predicted malignant'],
            yticklabels=['Actual benign', 'Actual malignant'])
plt.title('Confusion matrix')
plt.xlabel('Prediction')
plt.ylabel('Actual class')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'TN={tn}, FP={fp}, FN={fn}, TP={tp}')
sklearn_model = LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE)
sklearn_model.fit(X_train_scaled, y_train)

sklearn_scores = sklearn_model.predict_proba(X_test_scaled)[:, 1]
sklearn_predictions = sklearn_model.predict(X_test_scaled)
sklearn_metrics = show_metrics(y_test, sklearn_predictions, sklearn_scores, 'scikit-learn model')
coef_table = pd.DataFrame({
    'feature': X.columns,
    'coefficient': sklearn_model.coef_[0],
    'odds_multiplier': np.exp(sklearn_model.coef_[0])
}).sort_values('coefficient', key=np.abs, ascending=False)
display(coef_table.head(10).round(4))
print('Positive coefficients increase the log-odds of malignant (class 1).')
fpr, tpr, thresholds = roc_curve(y_test, sklearn_scores)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'Logistic regression (AUC={roc_auc_score(y_test, sklearn_scores):.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random classifier')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate / recall')
plt.title('ROC curve')
plt.legend()
plt.show()

thresholds_to_compare = [0.01, 1.00, 0.01]
threshold_rows = []
for threshold in thresholds_to_compare:
    pred = (sklearn_scores >= threshold).astype(int)
    threshold_rows.append({
        'threshold': threshold, 
        'precision': precision_score(y_test, pred, zero_division=0),
        'recall': recall_score(y_test, pred, zero_division=0),
        'f1': f1_score(y_test, pred, zero_division=0)
    })
display(pd.DataFrame(threshold_rows).round(4))