import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

def plot_clusters_2d(X, labels, save_path='results/clusters_2d.png'):
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Music Clusters (t-SNE)')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_scores(scores_df, save_path='results/clustering_scores.png'):
    plt.figure(figsize=(10, 6))
    for metric in scores_df.columns:
        plt.plot(scores_df.index, scores_df[metric], marker='o', label=metric)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Clustering Metrics vs K')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
 
