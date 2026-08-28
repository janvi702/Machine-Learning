import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

X = np.linspace(1, 10, 30)
noise = np.random.normal(0, 4, size=X.shape)
y = 32 + 6.2 * X + noise

data = pd.DataFrame({'Study Hours': X, 'Exam Score': y})
display(data.head())
print(f'Number of observations: {len(data)}')
print(data.describe().round(2))

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='royalblue', edgecolor='black', alpha=0.8)
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Study Hours versus Exam Score')
plt.show()

def fit_analytical(x, target):
    x_mean = np.mean(x)
    y_mean = np.mean(target)
    numerator = np.sum((x - x_mean) * (target - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return intercept, slope

b0_analytical, b1_analytical = fit_analytical(X, y)
y_pred_analytical = b0_analytical + b1_analytical * X

print(f'Intercept (b0): {b0_analytical:.4f}')
print(f'Slope (b1):     {b1_analytical:.4f}')
print(f'Equation: Score = {b0_analytical:.2f} + {b1_analytical:.2f} × Study Hours')

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label='Actual data', color='royalblue', edgecolor='black')
plt.plot(X, y_pred_analytical, label='Best-fit line', color='crimson', linewidth=2.5)
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Analytical Linear Regression')
plt.legend()
plt.show()

def fit_gradient_descent(x, target, learning_rate=0.01, iterations=3000):
    intercept, slope = 0.0, 0.0
    n = len(x)
    cost_history = []

    for _ in range(iterations):
        predictions = intercept + slope * x
        errors = target - predictions
        cost_history.append(np.mean(errors ** 2))

        intercept_gradient = (-2 / n) * np.sum(errors)
        slope_gradient = (-2 / n) * np.sum(x * errors)
        intercept -= learning_rate * intercept_gradient
        slope -= learning_rate * slope_gradient

    return intercept, slope, np.array(cost_history)

b0_gd, b1_gd, costs = fit_gradient_descent(X, y)
y_pred_gd = b0_gd + b1_gd * X

print(f'Gradient-descent intercept: {b0_gd:.4f}')
print(f'Gradient-descent slope:     {b1_gd:.4f}')
print(f'Final MSE:                  {costs[-1]:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(costs, color='darkorange')
axes[0].set_title('Convergence of Gradient Descent')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Mean Squared Error')

axes[1].scatter(X, y, color='royalblue', edgecolor='black', label='Actual data')
axes[1].plot(X, y_pred_gd, color='green', linewidth=2.5, label='Gradient-descent line')
axes[1].set_title('Fitted Regression Line')
axes[1].set_xlabel('Study Hours')
axes[1].set_ylabel('Exam Score')
axes[1].legend()
plt.tight_layout()
plt.show()

def regression_metrics(actual, predicted):
    residuals = actual - predicted
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}

results = pd.DataFrame({
    'Analytical': regression_metrics(y, y_pred_analytical),
    'Gradient Descent': regression_metrics(y, y_pred_gd)
}).T
display(results.round(4))

residuals = y - y_pred_analytical
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(y_pred_analytical, residuals, color='purple', edgecolor='black')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Predicted Score')
axes[0].set_ylabel('Residual')
axes[0].set_title('Residual Plot')

axes[1].scatter(y, y_pred_analytical, color='teal', edgecolor='black')
limits = [min(y.min(), y_pred_analytical.min()), max(y.max(), y_pred_analytical.max())]
axes[1].plot(limits, limits, 'r--', label='Ideal prediction')
axes[1].set_xlabel('Actual Score')
axes[1].set_ylabel('Predicted Score')
axes[1].set_title('Actual versus Predicted')
axes[1].legend()
plt.tight_layout()
plt.show()

def predict(x_new, intercept, slope):
    return intercept + slope * np.asarray(x_new)

new_hours = 7.5
predicted_score = predict(new_hours, b0_analytical, b1_analytical)
print(f'Predicted score for {new_hours} study hours: {predicted_score:.2f}')
