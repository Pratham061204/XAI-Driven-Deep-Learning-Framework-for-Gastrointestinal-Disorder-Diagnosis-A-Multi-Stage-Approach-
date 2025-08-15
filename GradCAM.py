import random

viz_model = models_ensemble[0]
target_layer = viz_model.features[-1]
grad_cam = GradCAM(model=viz_model, target_layer=target_layer)

#5 random images
num_images_to_show = 5
unique_classes = np.unique(y_true)
if len(unique_classes) < num_images_to_show:
    print(f"Warning: Only {len(unique_classes)} unique classes found in the test set. Showing all.")
    num_images_to_show = len(unique_classes)

target_classes = random.sample(list(unique_classes), num_images_to_show)
print(f"Will display one image from each of these classes: {[valid_classes[i] for i in target_classes]}\n")

samples_to_viz = []
found_classes = set()

dl_test_single = DataLoader(ds_test, batch_size=1, shuffle=True)

for x_sample, y_sample in dl_test_single:
    y_true_idx = y_sample.item()
    if y_true_idx in target_classes and y_true_idx not in found_classes:
        x_sample, y_sample = x_sample.to(device), y_sample.to(device)
        
        viz_model.eval()
        with torch.no_grad():
            output = viz_model(x_sample)
        pred_idx = output.argmax(1).item()

        # Storing the image, true label, and predicted label
        samples_to_viz.append({
            "image": x_sample,
            "true_idx": y_true_idx,
            "pred_idx": pred_idx
        })
        found_classes.add(y_true_idx)

    if len(samples_to_viz) == num_images_to_show:
        break

for sample in samples_to_viz:
    x_sample = sample["image"]
    true_idx = sample["true_idx"]
    pred_idx = sample["pred_idx"]

    true_class = valid_classes[true_idx]
    pred_class = valid_classes[pred_idx]

    original_img = to_pil_image(x_sample[0])

    # Generating Saliency Map
    saliency_map = generate_saliency_map(viz_model, x_sample.clone(), pred_idx)

    # Generating Grad-CAM Heatmap
    heatmap = grad_cam(x_sample, pred_idx)
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Generating Grad-CAM Overlay
    overlay = cv2.addWeighted(np.uint8(255 * original_img), 0.6, heatmap_color, 0.4, 0)

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    title_color = 'green' if true_class == pred_class else 'red'
    fig.suptitle(f"True: {true_class} | Predicted: {pred_class}", fontsize=16, color=title_color)

    axs[0].imshow(original_img)
    axs[0].set_title("1. Original Image")
    axs[0].axis('off')

    axs[1].imshow(saliency_map, cmap='hot')
    axs[1].set_title("2. Saliency Map")
    axs[1].axis('off')

    axs[2].imshow(heatmap_color)
    axs[2].set_title("3. Grad-CAM Heatmap")
    axs[2].axis('off')

    axs[3].imshow(overlay)
    axs[3].set_title("4. Grad-CAM Overlay")
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()