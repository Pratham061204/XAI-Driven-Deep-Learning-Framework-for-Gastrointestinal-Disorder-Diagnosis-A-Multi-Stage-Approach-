import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

# Hyperparameters
epochs    = 20
swa_start = 10
lr        = 1e-4
swa_lr    = 1e-5

def fine_tune_convnext_tiny(seed: int, num_classes: int) -> nn.Module:
    torch.manual_seed(seed)
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

base_model = fine_tune_convnext_tiny(0, len(valid_classes)).to(device)
optimizer  = optim.Adam(base_model.parameters(), lr=lr)
criterion  = nn.CrossEntropyLoss()

# SWA wrapper
swa_model  = AveragedModel(base_model)
swa_sched  = SWALR(optimizer, swa_lr=swa_lr)

for epoch in range(epochs):
    base_model.train()
    for x, y in dl_train:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = base_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    if epoch >= swa_start:
        swa_model.update_parameters(base_model)
        swa_sched.step()

swa_model.to(device)
update_bn(dl_train, swa_model, device=device)

swa_model.eval()
y_true, y_pred, y_scores = [], [], []
with torch.no_grad():
    for x, y in dl_test:
        x, y = x.to(device), y.to(device)
        out = swa_model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds)
        y_scores.extend(probs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)

# Classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=valid_classes, digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_classes)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

# ROC AUC (micro-average) curve

n_classes = len(valid_classes)
y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'Micro-average ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, lw=2, label=f'Micro-average PR (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Micro-average Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

torch.save(swa_model.state_dict(), "convnext_tiny_swa.pth")
print("Saved SWA model to convnext_tiny_swa.pth")
