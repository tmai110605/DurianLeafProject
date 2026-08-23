import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

import matplotlib.pyplot as plt
import numpy as np

from config import (
    TEST_ROOT,
    TEST_CSV,
    CHECKPOINT_DIR,
    RESULT_DIR,
    NUM_DISEASE_CLASSES,
    NUM_SEVERITY_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEVICE
)

from dataset import DurianLeafDataset
from utils import get_device

from mobilenetv3_custom import (
    MobileNetV2MultiTask,
    MobileNetV3SmallMultiTask,
    MobileNetV3LargeMultiTask
)

from resnet_custom import (
    ResNet18MultiTask,
    ResNet50MultiTask
)
from densenet_custom import DenseNet121MultiTask
from ticknet_custom import TickNetLargeMultiTask

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-task durian leaf disease model"
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
        help="Model architecture to evaluate"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth file"
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory containing best checkpoint"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=IMAGE_SIZE,
        help="Input image size"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate used when building model"
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


def resolve_checkpoint_path(args):
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)

        if not run_dir.exists():
            run_dir = CHECKPOINT_DIR / args.run_dir

        checkpoint_path = run_dir / f"best_{args.model}_multitask.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return checkpoint_path

    # Default case: find checkpoint in CHECKPOINT_DIR
    checkpoint_path = CHECKPOINT_DIR / f"best_{args.model}_multitask.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please provide --checkpoint or --run_dir."
        )

    return checkpoint_path


def clean_state_dict(state_dict):
    """
    Remove THOP buffers if they exist.
    """
    return {
        k: v for k, v in state_dict.items()
        if not k.endswith("total_ops") and not k.endswith("total_params")
    }


def plot_confusion_matrix(cm, class_names, save_path, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold"
            )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_metrics_json(save_path, metrics):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def main():
    args = parse_args()

    # device = get_device(DEVICE)
    # print(f"Using device: {device}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(args)
    print(f"Using checkpoint: {checkpoint_path}")

    result_dir = RESULT_DIR / f"eval_{args.model}_{checkpoint_path.parent.name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = DurianLeafDataset(
        root_dir=TEST_ROOT,
        csv_file=TEST_CSV,
        transform=test_transform,
        use_severity=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = build_model(
        model_name=args.model,
        dropout=args.dropout
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = clean_state_dict(checkpoint["model_state_dict"])

    model.load_state_dict(state_dict)
    model.eval()

    disease_true = []
    disease_pred = []
    severity_true = []
    severity_pred = []
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for images, disease_labels, severity_labels in test_loader:
            images = images.to(device)

            disease_outputs, severity_outputs = model(images)

            disease_predictions = disease_outputs.argmax(dim=1).cpu().numpy()
            severity_predictions = severity_outputs.argmax(dim=1).cpu().numpy()

            disease_pred.extend(disease_predictions)
            severity_pred.extend(severity_predictions)

            disease_true.extend(disease_labels.numpy())
            severity_true.extend(severity_labels.numpy())
    
    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time_per_sample = total_time / len(test_dataset)
    print(f"\nEvaluation time: {total_time:.4f} seconds")
    print(f"Average time per sample: {avg_time_per_sample:.6f} seconds")
    
    disease_names = [
        "healthy",
        "algal",
        "allocaridara_attack",
        "blight",
        "phomopsis"
    ]

    severity_names = [
        "healthy",
        "mild",
        "moderate",
        "severe"
    ]

    disease_report = classification_report(
        disease_true,
        disease_pred,
        target_names=disease_names,
        digits=4
    )

    severity_report = classification_report(
        severity_true,
        severity_pred,
        target_names=severity_names,
        digits=4
    )

    print("Disease Classification Report")
    print(disease_report)

    print("Severity Classification Report")
    print(severity_report)

    disease_acc = accuracy_score(disease_true, disease_pred)
    severity_acc = accuracy_score(severity_true, severity_pred)

    disease_precision_macro, disease_recall_macro, disease_f1_macro, _ = (
        precision_recall_fscore_support(
            disease_true,
            disease_pred,
            average="macro",
            zero_division=0
        )
    )

    severity_precision_macro, severity_recall_macro, severity_f1_macro, _ = (
        precision_recall_fscore_support(
            severity_true,
            severity_pred,
            average="macro",
            zero_division=0
        )
    )

    metrics = {
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "image_size": args.image_size,
        "batch_size": args.batch_size,

        "disease_accuracy": disease_acc,
        "disease_precision_macro": disease_precision_macro,
        "disease_recall_macro": disease_recall_macro,
        "disease_f1_macro": disease_f1_macro,

        "severity_accuracy": severity_acc,
        "severity_precision_macro": severity_precision_macro,
        "severity_recall_macro": severity_recall_macro,
        "severity_f1_macro": severity_f1_macro,
    }

    # If checkpoint contains Params/FLOPs, save them to metrics
    for key in [
        "total_params",
        "trainable_params",
        "flops",
        "flops_readable",
        "params_readable",
        "epoch",
        "val_loss",
        "val_disease_acc",
        "val_severity_acc"
    ]:
        if key in checkpoint:
            metrics[key] = checkpoint[key]

    report_path = result_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")

        f.write("Disease Classification Report\n")
        f.write(disease_report)

        f.write("\n\nSeverity Classification Report\n")
        f.write(severity_report)

    save_metrics_json(result_dir / "metrics.json", metrics)

    disease_cm = confusion_matrix(disease_true, disease_pred)
    severity_cm = confusion_matrix(severity_true, severity_pred)

    plot_confusion_matrix(
        disease_cm,
        disease_names,
        result_dir / "confusion_matrix_disease.png",
        "Disease Confusion Matrix"
    )

    plot_confusion_matrix(
        severity_cm,
        severity_names,
        result_dir / "confusion_matrix_severity.png",
        "Severity Confusion Matrix"
    )

    print(f"Saved report to: {report_path}")
    print(f"Saved metrics to: {result_dir / 'metrics.json'}")
    print(f"Saved confusion matrices to: {result_dir}")


if __name__ == "__main__":
    main()