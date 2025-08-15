import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

swa_model.eval()

BG_N = 16
bg_list, seen = [], 0
with torch.no_grad():
    for xb, _ in dl_train:
        bg_list.append(xb)
        seen += xb.size(0)
        if seen >= BG_N:
            break
background = torch.cat(bg_list, dim=0)[:BG_N]

TARGET_CLASS_NAME = 'Duodenal bulb'  # can be anything out of 22
print(f"Searching for an image from class: '{TARGET_CLASS_NAME}'...")

# Creating a new loader to iterate through single images easily
dl_test_single = DataLoader(ds_test, batch_size=1, shuffle=True)
x_one = None
true_y = None

try:
    target_class_idx = valid_classes.index(TARGET_CLASS_NAME)
except ValueError:
    raise ValueError(f"Error: Class '{TARGET_CLASS_NAME}' not found. Available classes are: {valid_classes}")

for img_tensor, label_tensor in dl_test_single:
    if label_tensor.item() == target_class_idx:
        x_one = img_tensor
        true_y = label_tensor.item()
        print(f"Found image with true label: '{valid_classes[true_y]}'")
        break

if x_one is None:
    raise RuntimeError(f"Could not find any images for class '{TARGET_CLASS_NAME}' in the test set.")

mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
def to_nhwc_denorm(x_nchw: torch.Tensor) -> np.ndarray:
    xd = (x_nchw.cpu() * std + mean).clamp(0, 1)
    return xd.permute(0, 2, 3, 1).numpy()

@torch.no_grad()
def predict_fn(nhwc_imgs: np.ndarray) -> np.ndarray:
    imgs = torch.from_numpy(nhwc_imgs).permute(0, 3, 1, 2).to(device).float()
    imgs = (imgs - mean.to(device)) / std.to(device)
    logits = swa_model(imgs)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

H, W = x_one.shape[2], x_one.shape[3]
masker = shap.maskers.Image("blur(32,32)", (H, W, 3))
explainer = shap.Explainer(predict_fn, masker, output_names=valid_classes)

# Deciding top 6 classes to explain
with torch.no_grad():
    p = torch.softmax(swa_model(x_one.to(device)), dim=1)[0].cpu().numpy()
TOPK = min(6, len(valid_classes))
topk_idx = np.argsort(-p)[:TOPK]

x_one_denorm = to_nhwc_denorm(x_one)
shap_values = explainer(x_one_denorm, max_evals=800, outputs=topk_idx)
def to_image_plot_inputs(explanation):
    vals = explanation.values
    if vals.ndim == 5:
        return [vals[..., i] for i in range(vals.shape[-1])]
    return [vals]
sv_for_plot = to_image_plot_inputs(shap_values)

col_labels = [valid_classes[i] for i in topk_idx]
labels_array = np.array([col_labels])
true_pred_text = f"True: {valid_classes[true_y]}   |   Pred: {valid_classes[int(p.argmax())]}"
true_labels_array = np.array([true_pred_text])

shap.image_plot(
    sv_for_plot,
    x_one_denorm,
    labels=labels_array,
    true_labels=true_labels_array,
    show=False
)
fig = plt.gcf()
fig.set_figwidth(5 * len(sv_for_plot) + 3)
fig.set_figheight(5)
all_axes = fig.get_axes()
if all_axes:
    for i, ax in enumerate(all_axes[1:-1]):
        ax.set_title(col_labels[i], fontsize=10)
    all_axes[0].set_ylabel(true_pred_text, fontsize=9)
    colorbar_ax = all_axes[-1]
    pos = colorbar_ax.get_position()
    colorbar_ax.set_position([pos.x0, 0.05, pos.width, pos.height])
plt.show()