# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style='whitegrid', context='notebook')
print('Libraries imported successfully.')

centers = [(28, 25), (32, 78), (68, 30), (72, 75)]
cluster_std = [6.5, 7.0, 7.5, 6.0]
X_raw, _ = make_blobs(
    n_samples=320, centers=centers, cluster_std=cluster_std,
    random_state=RANDOM_STATE
)
df = pd.DataFrame(X_raw, columns=['Annual Income (₹ lakh scaled)', 'Spending Score'])
df['Annual Income (₹ lakh scaled)'] = df['Annual Income (₹ lakh scaled)'].clip(10, 100)
df['Spending Score'] = df['Spending Score'].clip(1, 100)
print('Dataset shape:', df.shape)
display(df.head(10))

df.info()
print('\nMissing values:')
display(df.isna().sum().to_frame('Missing'))
print('Summary statistics:')
display(df.describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
sns.histplot(df.iloc[:, 0], kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Income Distribution')
sns.histplot(df.iloc[:, 1], kde=True, ax=axes[1], color='darkorange')
axes[1].set_title('Spending Score Distribution')
sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], ax=axes[2], s=50)
axes[2].set_title('Customers Before Clustering')
plt.tight_layout()
plt.show()

features = df.columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
scaled_df = pd.DataFrame(X_scaled, columns=features)
display(scaled_df.describe().round(3))

k_values = range(2, 11)
inertias, silhouette_scores = [], []
for k in k_values:
    model = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

evaluation = pd.DataFrame({
    'K': list(k_values), 'Inertia': inertias, 'Silhouette Score': silhouette_scores
})
display(evaluation.round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(list(k_values), inertias, marker='o')
axes[0].set(title='Elbow Method', xlabel='Number of clusters (K)', ylabel='Inertia')
axes[1].plot(list(k_values), silhouette_scores, marker='o', color='darkgreen')
axes[1].set(title='Silhouette Analysis', xlabel='Number of clusters (K)', ylabel='Silhouette score')
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.show()

best_k_by_silhouette = int(evaluation.loc[evaluation['Silhouette Score'].idxmax(), 'K'])
print('Best K by silhouette score:', best_k_by_silhouette)

FINAL_K = 4
kmeans = KMeans(
    n_clusters=FINAL_K, init='k-means++', n_init=20,
    max_iter=300, random_state=RANDOM_STATE
)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print('Iterations used:', kmeans.n_iter_)
print('Final inertia:', round(kmeans.inertia_, 3))
print('Silhouette score:', round(silhouette_score(X_scaled, df['Cluster']), 3))
display(df.head())

centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_original, columns=features)
centroids_df.index.name = 'Cluster'
display(centroids_df.round(2))

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df, x=features[0], y=features[1], hue='Cluster',
    palette='tab10', s=65, alpha=0.75
)
plt.scatter(
    centroids_original[:, 0], centroids_original[:, 1],
    c='black', marker='X', s=260, label='Centroids', edgecolors='white'
)
plt.title('K-Means Customer Segments')
plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

cluster_profile = df.groupby('Cluster').agg(
    Customers=('Cluster', 'size'),
    Mean_Income=(features[0], 'mean'),
    Mean_Spending=(features[1], 'mean')
).round(2)
cluster_profile['Percentage'] = (100 * cluster_profile['Customers'] / len(df)).round(1)
display(cluster_profile)

income_mid = df[features[0]].median()
spending_mid = df[features[1]].median()

def describe_segment(row):
    income = 'High-income' if row['Mean_Income'] >= income_mid else 'Lower-income'
    spending = 'high-spending' if row['Mean_Spending'] >= spending_mid else 'low-spending'
    return f'{income}, {spending}'

cluster_profile['Suggested Segment'] = cluster_profile.apply(describe_segment, axis=1)
display(cluster_profile[['Suggested Segment']])

df['Silhouette'] = silhouette_samples(X_scaled, df['Cluster'])
display(df.groupby('Cluster')['Silhouette'].agg(['mean', 'min', 'max']).round(3))

plt.figure(figsize=(9, 4.5))
sns.boxplot(data=df, x='Cluster', y='Silhouette', hue='Cluster', palette='tab10', legend=False)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Silhouette Values by Cluster')
plt.tight_layout()
plt.show()

new_customers = pd.DataFrame({
    features[0]: [25, 75, 30, 70],
    features[1]: [20, 80, 85, 25]
})
new_scaled = scaler.transform(new_customers)
new_customers['Predicted Cluster'] = kmeans.predict(new_scaled)
new_customers['Suggested Segment'] = new_customers['Predicted Cluster'].map(
    cluster_profile['Suggested Segment']
)
display(new_customers)
