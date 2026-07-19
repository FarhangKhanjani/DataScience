"""Generate portfolio charts from the fraud-detection notebook results.

All numbers below are taken directly from the notebook's actual outputs.
Usage:  python fraud_charts.py   ->  saves 3 PNGs to ./portfolio_images/
"""
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("portfolio_images")
out.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

# 1. KNN accuracy vs k (CV / train / test)
k = [5, 10, 15]
cv_acc   = [0.9991, 0.9990, 0.9989]
train_acc = [0.9993, 0.9991, 0.9991]
test_acc  = [0.9993, 0.9990, 0.9990]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(k, cv_acc, "o-", label="5-fold CV accuracy")
ax.plot(k, train_acc, "s-", label="Training accuracy")
ax.plot(k, test_acc, "^-", label="Testing accuracy")
ax.set_xticks(k)
ax.set_xlabel("n_neighbors (k)")
ax.set_ylabel("Accuracy")
ax.set_title("KNN Fraud Classifier — Accuracy vs. Number of Neighbors")
ax.legend()
fig.tight_layout()
fig.savefig(out / "knn_accuracy_vs_k.png", bbox_inches="tight")

# 2. K-Means silhouette analysis
clusters = [2, 3, 4]
sil = [0.9146, 0.8609, 0.8064]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar([str(c) for c in clusters], sil, color=["#2a9d8f", "#a8bdb8", "#a8bdb8"])
for b, s in zip(bars, sil):
    ax.text(b.get_x() + b.get_width()/2, s + 0.005, f"{s:.4f}", ha="center")
ax.set_ylim(0.75, 0.95)
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Average silhouette score")
ax.set_title("K-Means Silhouette Analysis — k=2 Best Separates Transactions")
fig.tight_layout()
fig.savefig(out / "kmeans_silhouette.png", bbox_inches="tight")

# 3. Lloyd vs Elkan efficiency benchmark
algos = ["Lloyd", "Elkan"]
train_t = [0.8028, 0.5313]
test_t = [0.0092, 0.0070]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
b0 = axes[0].bar(algos, train_t, color=["#8ab6d6", "#2a6f97"])
axes[0].set_title("Training time (s)")
for b, v in zip(b0, train_t):
    axes[0].text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center")
b1 = axes[1].bar(algos, test_t, color=["#8ab6d6", "#2a6f97"])
axes[1].set_title("Inference time (s)")
for b, v in zip(b1, test_t):
    axes[1].text(b.get_x() + b.get_width()/2, v + 0.0001, f"{v:.4f}", ha="center")
fig.suptitle("K-Means: Elkan ~34% Faster Training at Identical Silhouette (0.9146)")
fig.tight_layout()
fig.savefig(out / "kmeans_lloyd_vs_elkan.png", bbox_inches="tight")

print(f"Saved 3 charts to {out.resolve()}")
