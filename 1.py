# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Style settings for better visuals
plt.style.use('seaborn-darkgrid')
sns.set_palette('Set2')

# Load the dataset
data = pd.read_csv('customer_journey_data (1).csv')

# Data Preprocessing
data = data.dropna()

# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Dimensionality Reduction using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])

# Determining the optimal number of clusters using Silhouette Scores
sil_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
    kmeans.fit(pca_df)
    sil_scores.append(silhouette_score(pca_df, kmeans.labels_))

# Choose the optimal number of clusters based on the highest silhouette score
optimal_clusters = sil_scores.index(max(sil_scores)) + 2
print(f'Optimal Number of Clusters: {optimal_clusters}')

# K-means Clustering with optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(pca_df)
pca_df['Cluster'] = cluster_labels

# Calculating silhouette score
sil_score = silhouette_score(pca_df[['PCA1', 'PCA2']], cluster_labels)
print(f'Silhouette Score: {sil_score}')

# 1. Clustered Data Overview
clustered_data = data.copy()
clustered_data['Cluster'] = cluster_labels
print("\nClustered Data Sample:")
print(clustered_data.head())

# 2. Displaying centroids in PCA space
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['PCA1', 'PCA2'])
centroids['Cluster'] = centroids.index
print("\nCluster Centroids in PCA Space:")
print(centroids)

# 3a. Enhanced Scatter Plot with Centroids
plt.figure(figsize=(14, 10))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.7, edgecolor='w', linewidth=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=400, c='red', marker='X', edgecolor='black', linewidth=1.5, label='Centroids')
for i, (x, y) in enumerate(kmeans.cluster_centers_):
    plt.text(x, y, f'Cluster {i}', fontsize=12, weight='bold', color='black', ha='center', va='center')
plt.title('Customer Journey Clusters with Centroids', fontsize=18, weight='bold')
plt.xlabel('PCA1', fontsize=14)
plt.ylabel('PCA2', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 3b. Modified Bar Graph: Customer Count per Cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=clustered_data, palette='Set3')
plt.title('Customer Count per Cluster', fontsize=16, weight='bold')
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                       textcoords='offset points')
plt.show()

# 3c. Histogram: Session Duration Distribution per Cluster
plt.figure(figsize=(12, 6))
for cluster in clustered_data['Cluster'].unique():
    subset = clustered_data[clustered_data['Cluster'] == cluster]
    sns.histplot(subset['Session_Duration'], label=f'Cluster {cluster}', kde=True, bins=15, alpha=0.6)
plt.title('Session Duration Distribution by Cluster')
plt.xlabel('Session Duration')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 3d. Line Graph: Mean Feature Values per Cluster
cluster_means = clustered_data.groupby('Cluster').mean().T
cluster_means.plot(kind='line', figsize=(14, 8), marker='o')
plt.title('Mean Feature Values per Cluster')
plt.xlabel('Features')
plt.ylabel('Mean Value')
plt.grid(True)
plt.legend(title='Cluster')
plt.show()
