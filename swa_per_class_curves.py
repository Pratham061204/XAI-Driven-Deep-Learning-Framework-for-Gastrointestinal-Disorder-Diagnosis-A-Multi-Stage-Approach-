import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

swa_model.eval()
y_true, y_scores = [], []

with torch.no_grad():
    for x, y in dl_test:
        x, y = x.to(device), y.to(device)
        out = swa_model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        y_true.extend(y.cpu().numpy())
        y_scores.extend(probs)

y_true = np.array(y_true)
y_scores = np.array(y_scores)
n_classes = len(valid_classes)
y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curves
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    axes[0].plot(fpr, tpr, lw=1)
axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey', lw=1)
axes[0].set_title("Per-Class ROC Curves")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")

# PR Curves
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
    axes[1].plot(recall, precision, lw=1)
axes[1].set_title("Per-Class Precision-Recall Curves")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")

plt.tight_layout()
plt.show()
