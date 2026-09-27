"""
Day 3: evaluate the trained model honestly.

- Metrics come from the TEST set (never touched during training/tuning).
- The confidence threshold is chosen from the VAL set instead, and is a
  human judgment call (accuracy vs. coverage trade-off) - this script
  prints/saves a table; you edit CONFIDENCE_THRESHOLD in .env yourself.
- Grad-CAM is optional and off by default because it's slow.

Run:
    python training/evaluate.py
    python training/evaluate.py --gradcam        # also save Grad-CAM examples
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = "training/config.yaml") -> dict:
    with open(REPO_ROOT / path) as f:
        return yaml.safe_load(f)


def load_split_dataset(config: dict, split: str, class_names: list[str]):
    processed_dir = REPO_ROOT / config["data"]["processed_dir"]
    img_size = tuple(config["image"]["size"])
    return tf.keras.utils.image_dataset_from_directory(
        processed_dir / split,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=img_size,
        batch_size=config["train"]["batch_size"],
        shuffle=False,  # keep order stable so y_true/y_pred line up
    )


def predict_all(model, dataset):
    y_true, y_pred_probs = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_true.append(np.argmax(labels.numpy(), axis=1))
        y_pred_probs.append(probs)
    return np.concatenate(y_true), np.concatenate(y_pred_probs)


def save_confusion_matrix(y_true, y_pred, class_names: list[str], out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def threshold_search(y_true, y_pred_probs, candidates: list[float]) -> list[dict]:
    """For each candidate threshold, report coverage (% of predictions
    confident enough to answer) and accuracy on just those confident
    predictions. Pick the threshold yourself: a lower threshold answers
    more often but is wrong more often on the ones it does answer."""
    confidences = y_pred_probs.max(axis=1)
    predictions = y_pred_probs.argmax(axis=1)
    correct = predictions == y_true

    report = []
    for t in candidates:
        mask = confidences >= t
        coverage = float(mask.mean())
        accuracy_when_confident = float(correct[mask].mean()) if mask.any() else None
        report.append(
            {
                "threshold": t,
                "coverage": round(coverage, 4),
                "accuracy_when_confident": (
                    round(accuracy_when_confident, 4)
                    if accuracy_when_confident is not None
                    else None
                ),
            }
        )
    return report


def make_gradcam_heatmap(model, base_model, image_batch, class_index: int):
    """Grad-CAM against MobileNetV2's last conv layer.

    Note: base_model is nested inside `model` as a sub-model layer. In
    Keras 3, slicing a new tf.keras.Model out of a nested sub-model's
    graph (e.g. `Model(model.input, base_layer.input)`) raises
    "Output ... is not connected to inputs" even though the tensor looks
    valid - it's a known limitation of tracing through nested Functional
    models. The workaround is to call the preprocessing layer *instances*
    directly as plain functions (not build a sub-Model from them), which
    reproduces the exact same preprocessing without needing graph slicing.
    """
    last_conv_layer = base_model.get_layer("Conv_1")
    grad_model = tf.keras.Model(
        inputs=base_model.input, outputs=[last_conv_layer.output, base_model.output]
    )

    resize_layer = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Resizing))
    rescale_layer = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Rescaling))
    preprocessed = rescale_layer(resize_layer(image_batch))

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(preprocessed)
        # Re-apply the head to get class scores from base_output
        x = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                x = layer(x)
        class_score = x[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_output[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_examples(model, base_model, dataset, class_names, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_per_class: dict[str, int] = {}

    for images, labels in dataset.unbatch().batch(1).take(60):
        true_idx = int(np.argmax(labels.numpy()[0]))
        class_name = class_names[true_idx]
        if saved_per_class.get(class_name, 0) >= 2:
            continue

        try:
            heatmap = make_gradcam_heatmap(model, base_model, images, true_idx)
        except Exception as e:  # Grad-CAM is optional polish - never fail the run
            print(f"  Grad-CAM skipped for a {class_name} sample: {e}")
            continue

        img = images[0].numpy().astype("uint8")
        heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], img.shape[:2]).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img)
        axes[0].set_title(f"Original ({class_name})")
        axes[0].axis("off")
        axes[1].imshow(img)
        axes[1].imshow(heatmap_resized.squeeze(), cmap="jet", alpha=0.5)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        fig.tight_layout()

        idx = saved_per_class.get(class_name, 0)
        fig.savefig(out_dir / f"{class_name}_{idx}.png", dpi=150)
        plt.close(fig)
        saved_per_class[class_name] = idx + 1

    print(f"  Grad-CAM examples saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--gradcam", action="store_true", help="also save Grad-CAM examples")
    args = parser.parse_args()

    config = load_config(args.config)

    labels_path = REPO_ROOT / config["paths"]["labels_path"]
    with open(labels_path) as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(REPO_ROOT / config["paths"]["best_checkpoint"])

    # --- Test set: the honest numbers ---
    test_ds = load_split_dataset(config, "test", class_names)
    y_true, y_pred_probs = predict_all(model, test_ds)
    y_pred = y_pred_probs.argmax(axis=1)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm_path = REPO_ROOT / config["paths"]["confusion_matrix_out"]
    cm = save_confusion_matrix(y_true, y_pred, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # --- Validation set: threshold search only, never used for the headline metrics ---
    val_ds = load_split_dataset(config, "val", class_names)
    val_true, val_probs = predict_all(model, val_ds)
    threshold_report = threshold_search(
        val_true, val_probs, config["threshold_search"]["candidates"]
    )
    print("\nConfidence threshold search (on validation set):")
    for row in threshold_report:
        print(f"  {row}")
    print(
        "\nPick a threshold above and set CONFIDENCE_THRESHOLD in your API's .env. "
        "Lower = answers more often but is wrong more often when it does."
    )

    threshold_report_path = REPO_ROOT / config["paths"]["threshold_report_out"]
    threshold_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_report_path, "w") as f:
        json.dump(threshold_report, f, indent=2)

    # --- Save everything export.py needs ---
    metrics = {
        "test_accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_names,
        "threshold_search": threshold_report,
    }
    metrics_path = REPO_ROOT / "training" / "checkpoints" / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    if args.gradcam:
        # Find the MobileNetV2 sub-model inside our functional model.
        base_model = next(
            layer for layer in model.layers if layer.name.startswith("mobilenetv2")
        )
        print("\nGenerating Grad-CAM examples (this is slow)...")
        save_gradcam_examples(
            model, base_model, test_ds, class_names, REPO_ROOT / config["paths"]["gradcam_dir"]
        )

    print("\nNext: python training/export.py")


if __name__ == "__main__":
    main()
