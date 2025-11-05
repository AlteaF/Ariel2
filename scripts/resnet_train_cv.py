#!/usr/bin/env python3
"""
train_resnet50_cv.py

Usage example:
    python train_resnet50_cv.py --data-root /path/to/splits --epochs 10 --batch-size 32

Expect /path/to/splits to contain 5 directories (e.g. split1, split2, ..., split5),
each directory containing class subfolders (ImageFolder format).
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import models, transforms, datasets
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # mps support (Apple Silicon) - only available in recent PyTorch
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class RemappedImageFolder(Dataset):
    """
    Wraps torchvision.datasets.ImageFolder to remap class indices to a global mapping.
    This keeps class index ordering consistent across folds.
    """
    def __init__(self, image_folder: datasets.ImageFolder, global_class_to_idx: dict):
        self.root = image_folder.root
        self.samples = image_folder.samples  # list of (path, local_target)
        self.transform = image_folder.transform
        self.loader = image_folder.loader
        self.local_class_to_idx = image_folder.class_to_idx
        self.global_class_to_idx = global_class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, local_target = self.samples[idx]
        # map local_target -> local classname -> global index
        # find classname by reverse lookup
        # It's slightly inefficient but datasets are usually not huge relative to transforms
        classname = None
        for name, li in self.local_class_to_idx.items():
            if li == local_target:
                classname = name
                break
        if classname is None:
            raise ValueError("Classname not found for local target.")
        global_target = self.global_class_to_idx[classname]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, global_target

# -------------------------
# Training / Evaluation
# -------------------------

def train_one_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    preds = []
    labels = []
    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds.append(outputs.detach().cpu())
        labels.append(targets.detach().cpu())
    epoch_loss = running_loss / len(dataloader.dataset)
    preds = torch.cat(preds, dim=0)
    pred_labels = preds.argmax(dim=1).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    acc = accuracy_score(labels, pred_labels)
    return epoch_loss, acc, labels, pred_labels

def evaluate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            preds.append(outputs.cpu())
            labels.append(targets.cpu())
    epoch_loss = running_loss / len(dataloader.dataset)
    preds = torch.cat(preds, dim=0)
    pred_labels = preds.argmax(dim=1).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    acc = accuracy_score(labels, pred_labels)
    # compute per-class precision/recall/f1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred_labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, pred_labels, average="macro", zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, pred_labels, average="micro", zero_division=0
    )
    cm = confusion_matrix(labels, pred_labels)
    metrics = {
        "loss": epoch_loss,
        "accuracy": acc,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "confusion_matrix": cm,
        "labels": labels,
        "pred_labels": pred_labels,
    }
    return metrics

# -------------------------
# Main cross-validation flow
# -------------------------

def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[{datetime.now().isoformat()}] Using device: {device}")

    # Data transforms (ImageNet defaults)
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

        # Automatically detect all split directories inside data_root
    split_dirs = sorted(
        [
            os.path.join(args.data_root, d)
            for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
        ]
    )

    n_folds = len(split_dirs)
    if n_folds < 2:
        raise RuntimeError(
            f"Need at least 2 split directories inside {args.data_root} for cross-validation; found {n_folds}."
        )

    print(f"Detected {n_folds} split directories (leave-one-out cross-validation will use each once as validation):")
    for p in split_dirs:
        print("  -", p)


    # Build global class set (union of classes across splits) and mapping
    all_classnames = set()
    imagefolders = []
    for p in split_dirs:
        imgf = datasets.ImageFolder(p, transform=data_transform)
        imagefolders.append(imgf)
        all_classnames.update(imgf.classes)
    all_classnames = sorted(list(all_classnames))
    global_class_to_idx = {c: i for i, c in enumerate(all_classnames)}
    n_classes = len(all_classnames)
    print(f"Global classes ({n_classes}): {all_classnames}")

    # Wrap each ImageFolder with remapper
    remapped_datasets = [RemappedImageFolder(imgf, global_class_to_idx) for imgf in imagefolders]

    # cross-validation
    fold_metrics = []
    os.makedirs(args.output_dir, exist_ok=True)

    for fold_idx in range(len(remapped_datasets)):
        print(f"\n=== Fold {fold_idx+1}/{len(remapped_datasets)}: leave out {split_dirs[fold_idx]} ===")
        # build train (concat of all except fold_idx) and val (the left out one)
        train_dsets = [d for i, d in enumerate(remapped_datasets) if i != fold_idx]
        val_dset = remapped_datasets[fold_idx]
        train_dataset = ConcatDataset(train_dsets)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Build model
        model = models.resnet50(pretrained=True)
        # freeze all params
        for param in model.parameters():
            param.requires_grad = False
        # replace fc
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
        # Only fc params are trainable
        for param in model.fc.parameters():
            param.requires_grad = True
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

        best_val_macro_f1 = -1.0
        best_ckpt_path = os.path.join(args.output_dir, f"best_fold{fold_idx+1}.pth")
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_macro_f1": []}

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, _, _ = train_one_epoch(model, device, train_loader, criterion, optimizer)
            val_metrics = evaluate(model, device, val_loader, criterion)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])

            print(
                f"Fold {fold_idx+1} Epoch {epoch}/{args.epochs} | "
                f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
                f"Val loss: {val_metrics['loss']:.4f}, Val acc: {val_metrics['accuracy']:.4f}, Val macro-F1: {val_metrics['macro_f1']:.4f}"
            )

            # Save best model by macro F1
            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "fold": fold_idx,
                        "global_class_to_idx": global_class_to_idx,
                    },
                    best_ckpt_path,
                )
                print(f"  -> New best model saved to {best_ckpt_path} (macro F1 {best_val_macro_f1:.4f})")

        # load best checkpoint for final evaluation on val (optional but good)
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        final_val_metrics = evaluate(model, device, val_loader, criterion)
        # store metrics in fold_metrics
        fold_result = {
            "fold": fold_idx + 1,
            "split_path": split_dirs[fold_idx],
            "metrics": final_val_metrics,
            "history": history,
            "best_ckpt": best_ckpt_path if os.path.exists(best_ckpt_path) else None,
        }
        fold_metrics.append(fold_result)

        # Print per-class metrics for this fold
        print(f"\n>>> Fold {fold_idx+1} final validation metrics:")
        print(f"Accuracy: {final_val_metrics['accuracy']:.4f}")
        print(f"Macro precision/recall/f1: {final_val_metrics['macro_precision']:.4f} / {final_val_metrics['macro_recall']:.4f} / {final_val_metrics['macro_f1']:.4f}")
        print(f"Micro precision/recall/f1: {final_val_metrics['micro_precision']:.4f} / {final_val_metrics['micro_recall']:.4f} / {final_val_metrics['micro_f1']:.4f}")
        print("Per-class (precision / recall / f1 / support):")
        for idx, cname in enumerate(all_classnames):
            p = final_val_metrics["precision_per_class"][idx] if idx < len(final_val_metrics["precision_per_class"]) else 0.0
            r = final_val_metrics["recall_per_class"][idx] if idx < len(final_val_metrics["recall_per_class"]) else 0.0
            f = final_val_metrics["f1_per_class"][idx] if idx < len(final_val_metrics["f1_per_class"]) else 0.0
            s = final_val_metrics["support_per_class"][idx] if idx < len(final_val_metrics["support_per_class"]) else 0
            print(f"  {cname}: {p:.4f} / {r:.4f} / {f:.4f}  (support={s})")
        print("Confusion matrix:")
        print(final_val_metrics["confusion_matrix"])

    # After all folds: aggregate and print averages
    print("\n\n=== Cross-validation summary ===")
    # metrics to average: accuracy, macro_f1, macro_precision, macro_recall, micro_f1, etc.
    agg = defaultdict(list)
    for fm in fold_metrics:
        m = fm["metrics"]
        agg["accuracy"].append(m["accuracy"])
        agg["macro_f1"].append(m["macro_f1"])
        agg["macro_precision"].append(m["macro_precision"])
        agg["macro_recall"].append(m["macro_recall"])
        agg["micro_f1"].append(m["micro_f1"])
        agg["micro_precision"].append(m["micro_precision"])
        agg["micro_recall"].append(m["micro_recall"])
        # per-class arrays: pad if necessary then average across folds
        agg.setdefault("precision_per_class", []).append(m["precision_per_class"])
        agg.setdefault("recall_per_class", []).append(m["recall_per_class"])
        agg.setdefault("f1_per_class", []).append(m["f1_per_class"])
    def mean_list(l): return float(np.mean(l))
    print(f"Average accuracy: {mean_list(agg['accuracy']):.4f} (+/- {np.std(agg['accuracy']):.4f})")
    print(f"Average macro-F1: {mean_list(agg['macro_f1']):.4f} (+/- {np.std(agg['macro_f1']):.4f})")
    print(f"Average macro-precision: {mean_list(agg['macro_precision']):.4f}")
    print(f"Average macro-recall: {mean_list(agg['macro_recall']):.4f}")

    # average per-class metrics: make sure shapes consistent, pad shorter with zeros (if a class missing in a fold support=0)
    def pad_and_stack(list_of_arrays, target_len=n_classes):
        stacked = []
        for arr in list_of_arrays:
            arr = np.array(arr)
            if arr.shape[0] < target_len:
                # pad with zeros at the end
                arr = np.pad(arr, (0, target_len - arr.shape[0]), mode="constant", constant_values=0.0)
            stacked.append(arr[:target_len])
        return np.vstack(stacked)

    precision_stack = pad_and_stack(agg["precision_per_class"])
    recall_stack = pad_and_stack(agg["recall_per_class"])
    f1_stack = pad_and_stack(agg["f1_per_class"])
    avg_precision_per_class = precision_stack.mean(axis=0)
    avg_recall_per_class = recall_stack.mean(axis=0)
    avg_f1_per_class = f1_stack.mean(axis=0)

    print("\nAverage per-class metrics across folds:")
    for idx, cname in enumerate(all_classnames):
        print(f"  {cname}: precision={avg_precision_per_class[idx]:.4f}, recall={avg_recall_per_class[idx]:.4f}, f1={avg_f1_per_class[idx]:.4f}")

    # Save summary to a file
    summary_path = os.path.join(args.output_dir, "cv_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Cross-validation summary\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Data root: {args.data_root}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Class list: {all_classnames}\n")
        f.write("\nPer-fold results:\n")
        for fm in fold_metrics:
            f.write(f"Fold {fm['fold']} (left-out: {fm['split_path']})\n")
            m = fm["metrics"]
            f.write(f"  Accuracy: {m['accuracy']:.4f}\n")
            f.write(f"  Macro-F1: {m['macro_f1']:.4f}\n")
            f.write(f"  Confusion matrix:\n{m['confusion_matrix']}\n")
            f.write("\n")
        f.write("\nAverages:\n")
        f.write(f"Average accuracy: {mean_list(agg['accuracy']):.4f}\n")
        f.write(f"Average macro-F1: {mean_list(agg['macro_f1']):.4f}\n")
    print(f"\nSaved CV summary to {summary_path}")

# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 with leave-one-split-out CV (PyTorch)")
    parser.add_argument("--data-root", type=str, required=True, help="Root folder containing 5 split folders")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Where to save checkpoints and summary")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam)")
    parser.add_argument("--lr-step", type=int, default=5, help="LR scheduler step size (epochs)")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="LR scheduler gamma")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
