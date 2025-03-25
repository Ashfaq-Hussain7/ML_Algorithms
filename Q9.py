import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Convert to DataFrame
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# Display first few rows
df.head()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
def apply_dim_reduction(method, X, components):
    """Applies a dimensionality reduction technique to reduce features."""
    model = method(n_components=components)
    X_reduced = model.fit_transform(X, y) if method == LDA else model.fit_transform(X)
    return X_reduced, model


# Reduce to 2D
X_pca_2, pca_2 = apply_dim_reduction(PCA, X_scaled, 2)
X_lda_2, lda_2 = apply_dim_reduction(LDA, X_scaled, 2)  # ✅ LDA works here
X_tsne_2, tsne_2 = apply_dim_reduction(TSNE, X_scaled, 2)
X_svd_2, svd_2 = apply_dim_reduction(TruncatedSVD, X_scaled, 2)

# Reduce to 3D (❌ Remove LDA)
X_pca_3, pca_3 = apply_dim_reduction(PCA, X_scaled, 3)
X_tsne_3, tsne_3 = apply_dim_reduction(TSNE, X_scaled, 3)
X_svd_3, svd_3 = apply_dim_reduction(TruncatedSVD, X_scaled, 3)


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (X_reduced, title) in zip(axes.ravel(), 
                                  [(X_pca_2, "PCA"), (X_lda_2, "LDA"), 
                                   (X_tsne_2, "t-SNE"), (X_svd_2, "SVD")]):
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="viridis", edgecolor='k')
    ax.set_title(f"{title} (2D)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(*scatter.legend_elements(), title="Classes")

plt.tight_layout()
plt.show()


def evaluate_model(X_transformed, y):
    """Evaluates a classifier using cross-validation."""
    model = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_transformed, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

# Evaluate 2D features
pca_2_score = evaluate_model(X_pca_2, y)
lda_2_score = evaluate_model(X_lda_2, y)
tsne_2_score = evaluate_model(X_tsne_2, y)
svd_2_score = evaluate_model(X_svd_2, y)

# Evaluate 3D features
pca_3_score = evaluate_model(X_pca_3, y)
tsne_3_score = evaluate_model(X_tsne_3, y)
svd_3_score = evaluate_model(X_svd_3, y)


# Print Accuracy Scores
print("\nFeature Reduction to 2D:")
print(f"PCA:   {pca_2_score[0]:.4f} ± {pca_2_score[1]:.4f}")
print(f"LDA:   {lda_2_score[0]:.4f} ± {lda_2_score[1]:.4f}")
print(f"t-SNE: {tsne_2_score[0]:.4f} ± {tsne_2_score[1]:.4f}")
print(f"SVD:   {svd_2_score[0]:.4f} ± {svd_2_score[1]:.4f}")

print("\nFeature Reduction to 3D:")
print(f"PCA:   {pca_3_score[0]:.4f} ± {pca_3_score[1]:.4f}")
print(f"t-SNE: {tsne_3_score[0]:.4f} ± {tsne_3_score[1]:.4f}")
print(f"SVD:   {svd_3_score[0]:.4f} ± {svd_3_score[1]:.4f}")


