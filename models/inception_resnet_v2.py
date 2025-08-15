batch_size    = 128
num_epochs    = 10
learning_rate = 1e-4

def fine_tune_inceptionresnetv2(seed: int, num_classes: int) -> nn.Module:
    torch.manual_seed(seed)
    model = timm.create_model(
        "inception_resnet_v2",
        pretrained=True,
        num_classes=num_classes
    )
    return model

def train_model(model: nn.Module,
                dl_train: DataLoader,
                dl_val: DataLoader,
                epochs: int = num_epochs,
                lr: float = learning_rate) -> nn.Module:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc   = 0.0
    best_state = model.state_dict()

    for epoch in range(epochs):
        model.train()
        t_correct = t_total = 0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{epochs} Train")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            preds = out.argmax(1)
            t_correct += (preds == y).sum().item()
            t_total   += y.size(0)
            pbar.set_postfix(train_acc=f"{t_correct/t_total*100:5.2f}%")

        model.eval()
        v_correct = v_total = 0
        pbar = tqdm(dl_val, desc=f"Epoch {epoch+1}/{epochs} Val  ")
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(1)
                v_correct += (preds == y).sum().item()
                v_total   += y.size(0)
                pbar.set_postfix(val_acc=f"{v_correct/v_total*100:5.2f}%")

        val_acc = v_correct / v_total
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = model.state_dict()

    # restoring best validation model
    model.load_state_dict(best_state)
    return model

states, final_accs = [], []
for seed in range(5):
    print(f"\n=== Seed {seed} ===")
    m = fine_tune_inceptionresnetv2(seed, len(valid_classes))
    m = train_model(m, dl_train, dl_val)

    m.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dl_val:
            x, y = x.to(device), y.to(device)
            pred = m(x).argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    acc = correct / total
    print(f"Seed {seed} validation acc: {acc*100:5.2f}%")

    states.append(m.state_dict())
    final_accs.append(acc)

print("\nPer-seed validation accuracies:", [f"{a*100:5.2f}%" for a in final_accs])
