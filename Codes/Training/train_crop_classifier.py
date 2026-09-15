"""
train_crop_classifier.py

Trains a small MLP head on cached frozen-DINOv2 embeddings to classify
compliant vs. violation crops. Uses class-weighted loss to account for the
~3:1 helmet:head imbalance carried over from the original dataset.

Usage:
    python train_crop_classifier.py \
        --features_dir crop_features \
        --out_path stage2_classifier.pt \
        --epochs 30
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device, threshold=None):
    """If threshold is None, uses argmax (hard decision). Otherwise, flags
    class 1 (violation) whenever its softmax probability exceeds `threshold`
    -- this is what lets Arm 2 be evaluated at different operating points,
    the same way Arm 1's confidence threshold was swept.
    """
    model.eval()
    correct, total = 0, 0
    tp = fp = fn = 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            if threshold is None:
                preds = logits.argmax(dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
                preds = (probs[:, 1] > threshold).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    acc = correct / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return acc, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = torch.load(f"{args.features_dir}/train.pt")
    valid_data = torch.load(f"{args.features_dir}/valid.pt")
    test_data = torch.load(f"{args.features_dir}/test.pt")

    train_ds = TensorDataset(train_data["features"], train_data["labels"])
    valid_ds = TensorDataset(valid_data["features"], valid_data["labels"])
    test_ds = TensorDataset(test_data["features"], test_data["labels"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    in_dim = train_data["features"].shape[1]
    model = MLPHead(in_dim).to(device)

    # class weights: inverse frequency, to counter the ~3:1 helmet:head imbalance
    n_compliant = (train_data["labels"] == 0).sum().item()
    n_violation = (train_data["labels"] == 1).sum().item()
    total = n_compliant + n_violation
    weight = torch.tensor([
        total / (2 * n_compliant),
        total / (2 * n_violation),
    ], dtype=torch.float32).to(device)
    print(f"Class counts -> compliant: {n_compliant}, violation: {n_violation}")
    print(f"Class weights -> compliant: {weight[0]:.3f}, violation: {weight[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)

        val_acc, val_prec, val_rec = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={total_loss/len(train_ds):.4f} "
              f"| val_acc={val_acc:.4f} | val_violation_precision={val_prec:.4f} "
              f"| val_violation_recall={val_rec:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out_path)

    # final test-set evaluation using best checkpoint
    model.load_state_dict(torch.load(args.out_path))
    test_acc, test_prec, test_rec = evaluate(model, test_loader, device)
    print(f"\n=== Final test set (best val checkpoint, argmax decision) ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Violation precision: {test_prec:.4f}  (i.e. 1 - false-alarm rate)")
    print(f"Violation recall: {test_rec:.4f}  (i.e. 1 - missed-violation rate)")

    print(f"\n=== Threshold sweep on test set (for operating-point selection) ===")
    for t in [0.3, 0.5, 0.7, 0.9]:
        acc, prec, rec = evaluate(model, test_loader, device, threshold=t)
        print(f"  threshold={t}: precision={prec:.4f}, recall={rec:.4f}")


if __name__ == "__main__":
    main()
