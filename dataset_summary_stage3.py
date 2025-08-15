project_directory = "/content/drive/MyDrive/Gastrovision"

def get_valid_classes(root_dir, min_count=26):
    valid = []
    for cls in os.listdir(root_dir):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        count = len(glob(os.path.join(cls_dir, '*.jpg')))
        if count >= min_count:
            valid.append(cls)
    return sorted(valid)

valid_classes = get_valid_classes(project_directory)
print(f"Kept {len(valid_classes)} classes:", valid_classes)

class_image_summary = {}
total_images = 0

for cls in valid_classes:
    jpg_files = glob(os.path.join(project_directory, cls, '*.jpg'))
    class_image_summary[cls] = {
        'image_count': len(jpg_files),
        'sample_image': random.choice(jpg_files)
    }
    total_images += len(jpg_files)

print(f"Total images across kept classes: {total_images}")
for cls, info in class_image_summary.items():
    print(f"{cls}: {info['image_count']} images")
    img = Image.open(info['sample_image'])
    plt.figure(figsize=(3,3))
    plt.title(f"Sample from {cls}")
    plt.axis('off')
    plt.imshow(img)
    plt.show()