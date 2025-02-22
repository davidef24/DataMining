
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt





# Load the dataset
data = pd.read_csv('housing.csv')
# basic statistics
print(data.describe())
print(data.info())

# count missing values per column
missing_counts = data.isnull().sum()

# print summary of missing values
print("Missing Values Summary:")
print(missing_counts[missing_counts > 0])


# distribution of numerical variables
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[num_cols].hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.show()

# relationship between longitude, latitude, and house value
plt.figure(figsize=(12, 8))  # Adjust figure size for better fit
scatter = plt.scatter(
    data['longitude'], 
    data['latitude'], 
    c=data['median_house_value'], 
    cmap='coolwarm', 
    alpha=0.6
)
cbar = plt.colorbar(scatter, label='Median House Value')
cbar.ax.tick_params(labelsize=10)  # Adjust colorbar label size
plt.title("House Prices by Location", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)

# adjust layout
plt.tight_layout()
plt.show()

# correlation heatmap
plt.figure(figsize=(12, 8))
# select only numeric columns for the correlation matrix
numeric_data = data.select_dtypes(include=['float64', 'int64'])
# compute the correlation matrix
corr_matrix = numeric_data.corr()
# plot the heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Boxplot for ocean_proximity vs median_house_value
plt.figure(figsize=(12, 8))
sns.boxplot(x='ocean_proximity', y='median_house_value', data=data)
plt.title("House Value by Ocean Proximity")
plt.xticks(rotation=50)
plt.tight_layout()
plt.show()

# -------- Clustering on Raw Data --------

# Even if raw data, I need to process categorical variable. For the "raw" data, I used LabelEncoder while in the processing phase I use one-hot encoding
data_raw = data.copy()
label_encoder = LabelEncoder()
data_raw['ocean_proximity'] = label_encoder.fit_transform(data_raw['ocean_proximity'])
print(data_raw)
##
## drop rows with null values
print(f"Rows before dropping nulls: {data_raw.__len__()}")
data_raw = data_raw.dropna()
print(f"Rows after dropping nulls: {data_raw.__len__()}")
##
# apply standard scaler to center and normalize data for applying PCA and visualize clusters
scaler2 = StandardScaler()
scaled_data2 = scaler2.fit_transform(data_raw)
pca = PCA(n_components=3)  # Use 2 components
X_pca = pca.fit_transform(scaled_data2)

### Determine Silhouette Scores for different k values
silhouette_scores_raw = []
k_values = range(2, 11)  # Silhouette score is undefined for k=1

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++')
    cluster_labels = kmeans.fit_predict(data_raw)
    score = silhouette_score(data_raw, cluster_labels)
    silhouette_scores_raw.append(score)

best_score = max(silhouette_scores_raw)
print(f"The best silhouette score for NON processed data is {best_score:.4f}")

# Plot silhouette scores for raw data
plt.figure(figsize=(10, 5))
plt.plot(k_values, silhouette_scores_raw, marker='o')
plt.title('Silhouette Method for Raw Data')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Choose the optimal k based on the highest silhouette score
optimal_k_raw = k_values[np.argmax(silhouette_scores_raw)]
print(f"Optimal number of clusters for raw data: {optimal_k_raw}")

# Fit KMeans with the optimal number of clusters
# Measure running time for non-processed data
start_raw = time.time()
kmeans_raw = KMeans(n_clusters=optimal_k_raw, init='k-means++')
clusters_raw = kmeans_raw.fit_predict(data_raw)
end_raw = time.time()

time_raw = end_raw - start_raw
print(f"Running time for clustering on non-processed data: {time_raw:.4f} seconds")
#
data_raw_pca_kmeans = pd.concat([data_raw.reset_index(drop=True), pd.DataFrame(X_pca)], axis=1)
data_raw_pca_kmeans.columns.values[-3:]= ["pc1", "pc2", "pc3"]
data_raw_pca_kmeans["clusters"] = clusters_raw

# 3D Scatter plot for PCA components
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with cluster labels as hue
scatter = ax.scatter(
    data_raw_pca_kmeans["pc1"], 
    data_raw_pca_kmeans["pc2"], 
    data_raw_pca_kmeans["pc3"], 
    c=data_raw_pca_kmeans["clusters"], 
    cmap='tab10', 
    s=10, 
    alpha=0.6
)

ax.set_title(f"Clusters in PCA Space (k={optimal_k_raw})")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# Add legend for clusters
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

plt.show()

# plot clusters 
plt.figure(figsize=(10, 5))
sns.scatterplot(x='pc1', y='pc2', hue='clusters', data=data_raw_pca_kmeans, palette='magma')
plt.title(f'Clusters on Raw Data (k={optimal_k_raw}) plotted on PC1 and PC2')
plt.show()

# -------- Clustering on Processed Data --------

# Convert X_pca to a DataFrame

data_processed = data.copy()
data_processed = data_processed.dropna()
# various attempts in addition to scaling and one-hot encoding
#data_processed['rooms_per_household'] = data_processed['total_rooms'] / data_processed['households']
#data_processed['bedrooms_per_room'] = data_processed['total_bedrooms'] / data_processed['total_rooms']
#data_processed['population_per_household'] = data_processed['population'] / data_processed['households']
#data_processed['sqrt_population'] = np.sqrt(data_processed['population'])
#data_processed['households_per_population'] = data_processed['households'] / data_processed['population']
#data_processed['rooms_per_population'] = data_processed['total_rooms'] / data_processed['population']
#data_processed['bedrooms_per_households'] = data_processed['total_bedrooms'] / data_processed['households']
#data_processed['log_median_income'] = np.log1p(data_processed['median_income'])

#from sklearn.preprocessing import PolynomialFeatures
#poly = PolynomialFeatures(degree=2, include_bias=False)
#poly_features = poly.fit_transform(data_processed[['median_income', 'total_rooms']])
#poly_feature_names = poly.get_feature_names_out(['median_income', 'total_rooms'])
#
#poly_features_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=data_processed.index)
#
#data_processed = pd.concat([data_processed, poly_features_df], axis=1)

# apply one-hot encoding
ocean_proximity_encoded = pd.get_dummies(data_processed['ocean_proximity'], prefix='ocean_proximity')
data_processed = pd.concat([data_processed.drop('ocean_proximity', axis=1), ocean_proximity_encoded], axis=1)

print(data_processed)

# apply min-max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_processed)

# determine Silhouette Scores for different k values
silhouette_scores_raw2 = []
k_values = range(2, 10)  # Silhouette score is undefined for k=1

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++')
    cluster_labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores_raw2.append(score)

best_score = max(silhouette_scores_raw2)
print(f"The best silhouette score for processed data is {best_score:.4f}")

# Plot silhouette scores for raw data
plt.figure(figsize=(10, 5))
plt.plot(k_values, silhouette_scores_raw2, marker='o')
plt.title('Silhouette Method for Processed Data')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# choose the optimal k based on the highest silhouette score
optimal_k_raw2 = k_values[np.argmax(silhouette_scores_raw2)]
print(f"Optimal number of clusters for raw data: {optimal_k_raw2}")

# fit KMeans with the optimal number of clusters and measure time
start_processed = time.time()
kmeans_processed = KMeans(n_clusters=optimal_k_raw2, init='k-means++')
clusters_processed = kmeans_processed.fit_predict(scaled_data)
end_processed = time.time()


time_processed = end_processed - start_processed
print(f"Running time for clustering on processed data: {time_processed:.4f} seconds")


# apply standard scaler to center and normalize data for applying PCA and visualize clusters
scaler3 = StandardScaler()
scaled_data3 = scaler.fit_transform(scaled_data)
pca2 = PCA(n_components=5) 
X_pca2 = pca2.fit_transform(scaled_data3)
print(pca2.explained_variance_ratio_)
x_pca2_df = pd.DataFrame(X_pca2, columns=["pc1", "pc2", "pc3", "pc4", "pc5"])
x_pca2_df["clusters"] = clusters_processed

# 3D Scatter plot for PCA components
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with cluster labels as hue
scatter = ax.scatter(
    x_pca2_df["pc1"], 
    x_pca2_df["pc2"], 
    x_pca2_df["pc3"], 
    c=x_pca2_df["clusters"], 
    cmap='tab10', 
    s=10, 
    alpha=0.6
)

ax.set_title(f"Clusters in PCA Space (k={optimal_k_raw2})")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# Add legend for clusters
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

plt.show()
#