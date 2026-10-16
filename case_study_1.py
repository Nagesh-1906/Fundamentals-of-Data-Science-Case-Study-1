# ==============================================================================
# Fundamentals of Data Science - Case Study 1: Market Basket Clustering
# Author: Alapati Nagesh (Roll No: 160124733160)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# 1. Data Ingestion & Preprocessing
file_path = "Online retail_new.csv"

transactions = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        items = [item.strip().lower() for item in line.strip().split(",") 
                 if item.strip() and item.strip().lower() != "nan"]
        if items:
            transactions.append(items)

print(f"[INFO] Total Transactions Loaded: {len(transactions)}")

# 2. Descriptive Statistics & Exploratory Data Analysis
basket_sizes = [len(tx) for tx in transactions]
all_items = [item for tx in transactions for item in tx]
item_counts = Counter(all_items)
top_15 = item_counts.most_common(15)

print(f"Unique Items: {len(item_counts)}")
print(f"Mean Basket Size: {np.mean(basket_sizes):.2f}")
print(f"Median Basket Size: {np.median(basket_sizes):.1f}")
print(f"Std Deviation: {np.std(basket_sizes):.2f}")
print(f"Min / Max Basket Size: {np.min(basket_sizes)} / {np.max(basket_sizes)}")

# Plot 1: Top 15 Products
plt.figure(figsize=(10, 5))
df_top15 = pd.DataFrame(top_15, columns=["Product", "Frequency"])
sns.barplot(data=df_top15, x="Frequency", y="Product", palette="Blues_r")
plt.title("Figure 1: Top 15 Most Frequently Purchased Items", fontsize=13, weight="bold")
plt.xlabel("Frequency (Transaction Count)")
plt.ylabel("Catalog Product")
plt.tight_layout()
plt.savefig("figure1_top_products.png", dpi=300)
plt.close()

# Plot 2: Basket Size Distribution
plt.figure(figsize=(9, 4.5))
sns.histplot(basket_sizes, bins=range(1, 22), discrete=True, color="#1f77b4", edgecolor="black")
plt.title("Figure 2: Distribution of Basket Sizes Across Transactions", fontsize=13, weight="bold")
plt.xlabel("Number of Items per Basket")
plt.ylabel("Number of Transactions")
plt.xticks(range(1, 21))
plt.tight_layout()
plt.savefig("figure2_basket_size_distribution.png", dpi=300)
plt.close()

# 3. Vectorization & PCA
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(transactions)
print(f"[INFO] Binary Feature Matrix Shape: {X.shape}")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print(f"PC1 Variance: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"PC2 Variance: {pca.explained_variance_ratio_[1]*100:.2f}%")

# 4. Clustering Implementations (N=2500 Sample)
np.random.seed(42)
sample_idx = np.random.choice(len(X), size=2500, replace=False)
X_sub = X[sample_idx]
X_pca_sub = X_pca[sample_idx]

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_sub)
agg = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward').fit(X_sub)
dbscan = DBSCAN(eps=1.5, min_samples=15).fit(X_sub)
spectral = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', n_neighbors=10, random_state=42).fit(X_sub)

def get_metrics(data, labels, name):
    mask = labels != -1
    n_c = len(set(labels[mask]))
    if n_c > 1:
        sil = silhouette_score(data[mask], labels[mask])
        db = davies_bouldin_score(data[mask], labels[mask])
    else:
        sil, db = np.nan, np.nan
    return {
        "Algorithm": name, 
        "Clusters Found": n_c, 
        "Silhouette Score": round(sil, 4) if not np.isnan(sil) else "N/A", 
        "Davies-Bouldin Index": round(db, 4) if not np.isnan(db) else "N/A"
    }

results_df = pd.DataFrame([
    get_metrics(X_sub, kmeans.labels_, "K-Means (k=4)"),
    get_metrics(X_sub, agg.labels_, "Agglomerative (k=4)"),
    get_metrics(X_sub, dbscan.labels_, "DBSCAN"),
    get_metrics(X_sub, spectral.labels_, "Spectral (k=4)")
])
print("\n--- Clustering Benchmark Results ---")
print(results_df.to_string(index=False))

# Plot 3: 2D Cluster Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X_pca_sub[:, 0], X_pca_sub[:, 1], c=kmeans.labels_, cmap="tab10", s=18, alpha=0.7)
axes[0].set_title("Figure 3A: K-Means Clustering (k=4) in PCA Space", weight="bold")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

axes[1].scatter(X_pca_sub[:, 0], X_pca_sub[:, 1], c=agg.labels_, cmap="tab10", s=18, alpha=0.7)
axes[1].set_title("Figure 3B: Hierarchical Clustering (k=4) in PCA Space", weight="bold")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

plt.tight_layout()
plt.savefig("figure3_cluster_projections.png", dpi=300)
plt.close()
print("[INFO] Execution finished successfully. Figures saved.")