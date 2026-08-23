import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from training.config import (
    TRAIN_ROOT,
    VAL_ROOT,
    TRAIN_CSV,
    VAL_CSV,
    CHECKPOINT_DIR,
    NUM_DISEASE_CLASSES,
    NUM_SEVERITY_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE
)

from training.dataset import DurianLeafDataset
from training.utils import set_seed, get_device, count_parameters, calculate_flops

# Custom models
from models.mobilenetv3_custom import (
    MobileNetV2MultiTask,
    MobileNetV3SmallMultiTask,
    MobileNetV3LargeMultiTask
)

from models.resnet_custom import (
    ResNet18MultiTask,
    ResNet50MultiTask
)
from models.densenet_custom import DenseNet121MultiTask
from models.ticknet_custom import TickNetLargeMultiTask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multi-task durian leaf disease model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "mobilenetv2",
            "mobilenetv3_small",
            "mobilenetv3_large",
            "resnet18",
            "resnet50",
            "densenet121",
            "ticknet_large"
        ],
        help="Model architecture to train"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=IMAGE_SIZE,
        help="Input image size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate"
    )

    parser.add_argument(
        "--severity_loss_weight",
        type=float,
        default=1.0,
        help="Weight for severity loss"
    )

    parser.add_argument(
        "--use_severity_weight",
        action="store_true",
        help="Use class-weighted CrossEntropyLoss for severity"
    )

    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Save checkpoint for every epoch"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers"
    )

    return parser.parse_args()


def build_model(model_name, dropout):
    if model_name == "mobilenetv2":
        return MobileNetV2MultiTask(
            num_disease_classes=NUM_DISEASE_CLASSES,
            num_severity_classes=NUM_SEVERITY_CLASSES,
            dropout=dropout
        )

    if model_name == "mobilenetv3_small":
        return MobileNetV3SmallMultiTask(
            num_disease_classes=NUM_DISEASE_CLASSES,
            num_severity_classes=NUM_SEVERITY_CLASSES,
            dropout=dropout
        )

    if model_name == "mobilenetv3_large":
        return MobileNetV3LargeMultiTask(
            num_disease_classes=NUM_DISEASE_CLASSES,
            num_severity_classes=NUM_SEVERITY_CLASSES,
            dropout=dropout
        )

    if model_name == "resnet18":
        return ResNet18MultiTask(
            num_disease_classes=NUM_DISEASE_CLASSES,
            num_severity_classes=NUM_SEVERITY_CLASSES,
            dropout=dropout
        )

    if model_name == "resnet50":
        return ResNet50MultiTask(
            num_disease_classes=NUM_DISEASE_CLASSES,
            num_severity_classes=NUM_SEVERITY_CLASSES,
            dropout=dropout
        )
    if model_name == "densenet121":
        return DenseNet121MultiTask(
            num_disease_classes=NUM_DISEASE_CLASSES,
            num_severity_classes=NUM_SEVERITY_CLASSES,
            dropout=dropout
        )

    if model_name == "ticknet_large":
        return TickNetLargeMultiTask(
            num_disease_classes=NUM_DISEASE_CLASSES,
            num_severity_classes=NUM_SEVERITY_CLASSES,
            dropout=dropout
        )

    raise ValueError(f"Unsupported model: {model_name}")


def remove_thop_buffers(model):
    """
    THOP may add total_ops and total_params buffers to modules.
    Remove them before saving model.state_dict().
    """
    for module in model.modules():
        if hasattr(module, "total_ops"):
            delattr(module, "total_ops")
        if hasattr(module, "total_params"):
            delattr(module, "total_params")


def build_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def get_severity_weights(device):
    train_df = pd.read_csv(TRAIN_CSV)

    train_df["severity"] = pd.to_numeric(
        train_df["severity"],
        errors="coerce"
    )

    if train_df["severity"].isna().any():
        bad_rows = train_df[train_df["severity"].isna()]
        raise ValueError(
            "Found missing/invalid severity values in TRAIN_CSV.\n"
            f"{bad_rows[['file_name', 'file_path', 'disease_type', 'severity']].head(20)}"
        )

    train_df["severity"] = train_df["severity"].astype(int)

    severity_counts = train_df["severity"].value_counts().sort_index()

    severity_class_counts = torch.tensor(
        [severity_counts.get(i, 0) for i in range(NUM_SEVERITY_CLASSES)],
        dtype=torch.float
    )

    severity_class_counts = torch.clamp(severity_class_counts, min=1.0)

    severity_weights = 1.0 / severity_class_counts
    severity_weights = severity_weights / severity_weights.sum() * NUM_SEVERITY_CLASSES
    severity_weights = severity_weights.to(device)

    return severity_class_counts, severity_weights


def save_json_log(log_path, history):
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def main():
    args = parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_name = (
        f"{args.model}"
        f"_img{args.image_size}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr}"
        f"_sw{args.severity_loss_weight}"
    )

    if args.use_severity_weight:
        run_name += "_sevweight"

    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name: {run_name}")
    print(f"Checkpoint directory: {run_dir}")

    train_transform, val_transform = build_transforms(args.image_size)

    train_dataset = DurianLeafDataset(
        root_dir=TRAIN_ROOT,
        csv_file=TRAIN_CSV,
        transform=train_transform,
        use_severity=True
    )

    val_dataset = DurianLeafDataset(
        root_dir=VAL_ROOT,
        csv_file=VAL_CSV,
        transform=val_transform,
        use_severity=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = build_model(
        model_name=args.model,
        dropout=args.dropout
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    flops, thop_params, flops_readable, params_readable = calculate_flops(
        model=model,
        input_size=(1, 3, args.image_size, args.image_size),
        device=device
    )

    remove_thop_buffers(model)

    print("=" * 60)
    print("Model complexity")
    print(f"Model:                {args.model}")
    print(f"Input size:           {args.image_size}x{args.image_size}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"THOP parameters:      {params_readable}")
    print(f"FLOPs:                {flops_readable}")
    print("=" * 60)

    criterion_disease = nn.CrossEntropyLoss()

    if args.use_severity_weight:
        severity_class_counts, severity_weights = get_severity_weights(device)

        print("Severity class counts:", severity_class_counts.cpu().tolist())
        print("Severity class weights:", severity_weights.detach().cpu().tolist())

        criterion_severity = nn.CrossEntropyLoss(weight=severity_weights)
    else:
        criterion_severity = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_model_path = run_dir / f"best_{args.model}_multitask.pth"

    history = {
        "run_name": run_name,
        "model": args.model,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "severity_loss_weight": args.severity_loss_weight,
        "use_severity_weight": args.use_severity_weight,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "flops": int(flops),
        "flops_readable": flops_readable,
        "params_readable": params_readable,
        "epochs_log": []
    }

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()

        train_loss = 0.0
        train_disease_correct = 0
        train_severity_correct = 0
        train_total = 0

        loop = tqdm(
            train_loader,
            desc=f"{args.model} | Epoch {epoch + 1}/{args.epochs}"
        )

        for images, disease_labels, severity_labels in loop:
            images = images.to(device)
            disease_labels = disease_labels.to(device)
            severity_labels = severity_labels.to(device)

            optimizer.zero_grad()

            disease_outputs, severity_outputs = model(images)

            loss_disease = criterion_disease(
                disease_outputs,
                disease_labels
            )

            loss_severity = criterion_severity(
                severity_outputs,
                severity_labels
            )

            loss = loss_disease + args.severity_loss_weight * loss_severity

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            disease_preds = disease_outputs.argmax(dim=1)
            severity_preds = severity_outputs.argmax(dim=1)

            train_disease_correct += (disease_preds == disease_labels).sum().item()
            train_severity_correct += (severity_preds == severity_labels).sum().item()
            train_total += images.size(0)

            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_dataset)
        train_disease_acc = train_disease_correct / train_total
        train_severity_acc = train_severity_correct / train_total

        model.eval()

        val_loss = 0.0
        val_disease_correct = 0
        val_severity_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, disease_labels, severity_labels in val_loader:
                images = images.to(device)
                disease_labels = disease_labels.to(device)
                severity_labels = severity_labels.to(device)

                disease_outputs, severity_outputs = model(images)

                loss_disease = criterion_disease(
                    disease_outputs,
                    disease_labels
                )

                loss_severity = criterion_severity(
                    severity_outputs,
                    severity_labels
                )

                loss = loss_disease + args.severity_loss_weight * loss_severity

                val_loss += loss.item() * images.size(0)

                disease_preds = disease_outputs.argmax(dim=1)
                severity_preds = severity_outputs.argmax(dim=1)

                val_disease_correct += (disease_preds == disease_labels).sum().item()
                val_severity_correct += (severity_preds == severity_labels).sum().item()
                val_total += images.size(0)

        val_loss /= len(val_dataset)
        val_disease_acc = val_disease_correct / val_total
        val_severity_acc = val_severity_correct / val_total

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_disease_acc": train_disease_acc,
            "train_severity_acc": train_severity_acc,
            "val_loss": val_loss,
            "val_disease_acc": val_disease_acc,
            "val_severity_acc": val_severity_acc
        }

        history["epochs_log"].append(epoch_log)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Disease Acc: {train_disease_acc:.4f} | "
            f"Train Severity Acc: {train_severity_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Disease Acc: {val_disease_acc:.4f} | "
            f"Val Severity Acc: {val_severity_acc:.4f}"
        )

        checkpoint_data = {
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,

            "train_loss": train_loss,
            "train_disease_acc": train_disease_acc,
            "train_severity_acc": train_severity_acc,

            "val_loss": val_loss,
            "val_disease_acc": val_disease_acc,
            "val_severity_acc": val_severity_acc,

            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops": flops,
            "flops_readable": flops_readable,
            "params_readable": params_readable,

            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "severity_loss_weight": args.severity_loss_weight,
            "use_severity_weight": args.use_severity_weight
        }

        if args.save_all:
            epoch_model_path = run_dir / f"{args.model}_epoch_{epoch + 1:03d}.pth"
            torch.save(checkpoint_data, epoch_model_path)
            print(f"Saved epoch checkpoint: {epoch_model_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_data, best_model_path)
            print(f"Saved best model: {best_model_path}")

        save_json_log(run_dir / "training_log.json", history)

    total_time = time.time() - start_time
    history["total_training_time_seconds"] = total_time
    save_json_log(run_dir / "training_log.json", history)

    print("=" * 60)
    print("Training finished")
    print(f"Best model saved at: {best_model_path}")
    print(f"Training log saved at: {run_dir / 'training_log.json'}")
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()