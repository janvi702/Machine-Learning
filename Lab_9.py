import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print('Dataset shape:', X.shape)
print('Target: disease progression score (higher more progression)')
X.head()
# Split data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
print('Training samples:', X_train.shape[0])
print('Testing samples:', X_test.shape[0])
# Define models
models = {
'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
'Random Forest Regressor': RandomForestRegressor(n_estimators=200, random_state=42)
}
predictions = {}
for name, model in models.items():
model.fit(X_train, y_train)
predictions[name] = model.predict(X_test)
print(name, 'trained')
# Evaluate models
rows = []
for name, y_pred in predictions.items():
rows.append({
'Model': name,
'MAE': round(mean_absolute_error(y_test, y_pred), 2),
'MSE': round(mean_squared_error(y_test, y_pred), 2),
'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
'R2 Score': round(r2_score(y_test, y_pred), 4)
})
metrics_table = pd.DataFrame(rows).set_index('Model')
print(metrics_table)
# Cross-validation
for name, model in models.items():
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f'{name:26} CV R2: {scores.mean():.4f} (+/- {scores.std():.4f})')
# Select best model and plot
best_name = metrics_table['R2 Score'].idxmax()
y_best = predictions[best_name]
print('Best model by R2:', best_name)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(y test, y best, alpha=0.7, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title(f'Actual vs Predicted - {best_name}')
residuals = y_test - y_best
axes[1].scatter(y_best, residuals, alpha=0.7, color='darkorange')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Residual')
axes[1].set_title('Residual Plot')
plt.tight_layout()
plt.show()
best = metrics_table.loc[best_name]
