import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def to_pil_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    t = tensor.clone().detach().cpu()
    for channel, m, s in zip(t, mean, std):
        channel.mul_(s).add_(m)
    img_np = t.permute(1, 2, 0).numpy()
    return (img_np * 255).astype(np.uint8)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)
        for i in range(pooled_gradients.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

viz_model = swa_model.to(device)
dl_test_random = DataLoader(ds_test, batch_size=1, shuffle=True)
x_sample, y_sample = next(iter(dl_test_random))
x_sample, y_sample = x_sample.to(device), y_sample.to(device)

target_layer = viz_model.module.features[-1]
grad_cam_explainer = GradCAM(model=viz_model, target_layer=target_layer)
heatmap = grad_cam_explainer(x_sample)

original_img = to_pil_image(x_sample[0])
original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
max_index = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
center_y, center_x = max_index
radius = 50
color = (0, 255, 0)
thickness = 2
image_with_circle = cv2.circle(original_img_bgr.copy(), (center_x, center_y), radius, color, thickness)

with torch.no_grad():
    pred_idx = viz_model(x_sample).argmax(1).item()
true_class = valid_classes[y_sample.item()]
pred_class = valid_classes[pred_idx]

image_with_circle_rgb = cv2.cvtColor(image_with_circle, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(image_with_circle_rgb)
plt.title(f"Region of Interest\nTrue: {true_class} | Predicted: {pred_class}", fontsize=14)
plt.axis('off')
plt.show()
