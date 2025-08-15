class GastroVisionDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[cls]
        return img, label


all_samples = []
for cls in valid_classes:
    for p in glob(os.path.join(project_directory, cls, '*.jpg')):
        all_samples.append((p, cls))

train_s, temp_s = train_test_split(
    all_samples,
    test_size=0.30,
    stratify=[c for _, c in all_samples],
    random_state=42
)

class_to_idx = {c:i for i,c in enumerate(valid_classes)}

val_s, test_s = train_test_split(
    temp_s,
    test_size=0.50,
    stratify=[c for _, c in temp_s],
    random_state=42
)


train_transform = T.Compose([
    T.Resize((224,224)), T.RandomHorizontalFlip(),
    T.RandomRotation(25), T.ColorJitter(0.2,0.2),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = T.Compose([
    T.Resize((224,224)), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

batch_size = 32
ds_train = GastroVisionDataset(train_s, class_to_idx, train_transform)
ds_val   = GastroVisionDataset(val_s,   class_to_idx, val_transform)
ds_test  = GastroVisionDataset(test_s,  class_to_idx, val_transform)

dl_train = DataLoader(ds_train, batch_size, shuffle=True, num_workers=8)
dl_val   = DataLoader(ds_val,   batch_size, shuffle=False, num_workers=8)
dl_test  = DataLoader(ds_test,  batch_size, shuffle=False, num_workers=8)
