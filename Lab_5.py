# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from IPython.display import display
except ImportError:
    display = print

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

RANDOM_STATE = 42
sns.set_theme(style='whitegrid')

#1. Load and Inspect the Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')
target_names = iris.target_names

data = X.copy()
data['target'] = y
data['class_name'] = y.map(dict(enumerate(target_names)))

print('Feature matrix shape:', X.shape)
print('Number of classes:', len(target_names))
display(data.head())
display(data.groupby('class_name').size().rename('count'))
# Visualize two features and observe class separation
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='petal length (cm)', y='petal width (cm)',
                hue='class_name', palette='Set1', s=80)
plt.title('Iris samples using two features')
plt.show()

#2. Split the Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print('Training samples:', len(X_train))
print('Testing samples :', len(X_test))


# 3. Train the Decision Tree Classifier
tree_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=RANDOM_STATE
)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)
print('Test accuracy:', round(accuracy_score(y_test, y_pred), 4))
# Visualize the learned tree
plt.figure(figsize=(18, 9))
plot_tree(tree_model, feature_names=iris.feature_names,
          class_names=target_names, filled=True, rounded=True,
          impurity=True, proportion=True)
plt.title('Decision Tree for Multiclass Iris Classification')
plt.show()

#4. Evaluate the Model
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()
# Inspect feature importance
importance = pd.Series(tree_model.feature_importances_, index=iris.feature_names)
importance = importance.sort_values(ascending=False)
display(importance.to_frame('importance'))
importance.plot(kind='bar', color='teal', figsize=(8, 4))
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()


#5. Study Underfitting and Overfitting
depths = range(1, 11)
train_scores, test_scores = [], []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, marker='o', label='Training accuracy')
plt.plot(depths, test_scores, marker='s', label='Testing accuracy')
plt.xlabel('Maximum tree depth')
plt.ylabel('Accuracy')
plt.xticks(list(depths))
plt.legend()
plt.title('Effect of Tree Depth')
plt.show()

#6. Hyperparameter Tuning with Cross-Validation
parameter_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_split': [2, 4, 6]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    parameter_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train, y_train)

print('Best parameters:', grid.best_params_)
print('Best cross-validation accuracy:', round(grid.best_score_, 4))
tuned_pred = grid.predict(X_test)
print('Tuned model test accuracy:', round(accuracy_score(y_test, tuned_pred), 4))


#7. Predict a New Sample
new_sample = pd.DataFrame([[5.8, 2.8, 4.8, 1.4]], columns=iris.feature_names)
predicted_id = grid.predict(new_sample)[0]
predicted_probabilities = grid.predict_proba(new_sample)[0]

print('Predicted class:', target_names[predicted_id])
print('Class probabilities:')
display(pd.Series(predicted_probabilities, index=target_names).round(4))
