import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

print('Scikit-learn version:', sklearn.__version__)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

df = X.copy()
df['target'] = y
df['species'] = y.map(dict(enumerate(iris.target_names)))

print('Feature names:', list(X.columns))
print('Target names:', list(iris.target_names))
print('Dataset shape:', df.shape)
display(df.head())

print('Data types and missing values:')
display(pd.DataFrame({'dtype': X.dtypes, 'missing': X.isna().sum()}))

print('\nDescriptive statistics:')
display(X.describe().round(2))

print('\nClass distribution:')
display(df['species'].value_counts())

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)',
                hue='species', style='species', s=90)
plt.title('Iris flowers by petal measurements')
plt.grid(alpha=0.2)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print('Training samples:', X_train.shape[0])
print('Testing samples:', X_test.shape[0])
print('Training class counts:', y_train.value_counts().sort_index().to_dict())
print('Testing class counts:', y_test.value_counts().sort_index().to_dict())

logistic_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

logistic_model.fit(X_train, y_train)
y_pred_lr = logistic_model.predict(X_test)

print('Accuracy:', round(accuracy_score(y_test, y_pred_lr), 4))
print('\nClassification report:')
print(classification_report(y_test, y_pred_lr, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('Logistic Regression — Confusion Matrix')
plt.show()

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
    ])
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Macro Precision': precision_score(y_test, pred, average='macro'),
        'Macro Recall': recall_score(y_test, pred, average='macro'),
        'Macro F1': f1_score(y_test, pred, average='macro')
    })

results_df = pd.DataFrame(results).sort_values('Macro F1', ascending=False)
display(results_df.round(4))

plot_df = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 5))
sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric')
plt.ylim(0.80, 1.02)
plt.title('Test-set performance comparison')
plt.xticks(rotation=15)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

cv_results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_results.append({
        'Model': name,
        'Fold scores': np.round(scores, 3),
        'Mean CV accuracy': scores.mean(),
        'Std. deviation': scores.std()
    })

cv_df = pd.DataFrame(cv_results).sort_values('Mean CV accuracy', ascending=False)
display(cv_df)

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

parameter_grid = {
    'classifier__n_neighbors': list(range(1, 16)),
    'classifier__weights': ['uniform', 'distance'],
    'classifier__p': [1, 2]  # 1 = Manhattan; 2 = Euclidean distance
}

grid_search = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=parameter_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print('Best parameters:', grid_search.best_params_)
print('Best cross-validation accuracy:', round(grid_search.best_score_, 4))

best_knn = grid_search.best_estimator_
y_pred_tuned = best_knn.predict(X_test)
print('Tuned KNN test accuracy:', round(accuracy_score(y_test, y_pred_tuned), 4))

tree_model = models['Decision Tree']
plt.figure(figsize=(14, 7))
plot_tree(tree_model, feature_names=X.columns, class_names=iris.target_names,
          filled=True, rounded=True, fontsize=9)
plt.title('Decision Tree learned from the training data')
plt.show()

new_flowers = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2],
     [6.0, 2.9, 4.5, 1.5],
     [6.7, 3.1, 5.6, 2.4]],
    columns=X.columns
)

predicted_ids = best_knn.predict(new_flowers)
probabilities = best_knn.predict_proba(new_flowers)

prediction_table = new_flowers.copy()
prediction_table['predicted_species'] = iris.target_names[predicted_ids]
for index, species in enumerate(iris.target_names):
    prediction_table[f'P({species})'] = probabilities[:, index]

display(prediction_table.round(3))
