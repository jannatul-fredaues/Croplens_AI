"""
Day 2: train the classifier.

Two phases:
  1. Base frozen - train only the new classification head.
  2. Fine-tune - unfreeze the top layers of MobileNetV2 at a low LR.

Resizing, augmentation, and normalization are built as layers INSIDE the
model (see build_model), not done separately in a tf.data pipeline. Keras
preprocessing layers such as RandomFlip only apply when `training=True`,
which model.fit() sets automatically - so they are already a no-op at
inference, and the exact same model file used here is safe to export and
serve directly. This is what prevents train/serve preprocessing drift.

Run (locally or in Colab, after prepare_data.py):
    python training/train.py --config training/config.yaml
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf
import yaml
from tensorflow.keras import layers

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_datasets(config: dict):
    processed_dir = REPO_ROOT / config["data"]["processed_dir"]
    img_size = tuple(config["image"]["size"])
    batch_size = config["train"]["batch_size"]
    seed = config["seed"]

    def _load(split: str, shuffle: bool):
        return tf.keras.utils.image_dataset_from_directory(
            processed_dir / split,
            labels="inferred",
            label_mode="categorical",
            class_names=config["data"]["classes"],  # fixes label order explicitly
            image_size=img_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    train_ds = _load("train", shuffle=True)
    val_ds = _load("val", shuffle=False)
    test_ds = _load("test", shuffle=False)

    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names


def compute_class_weight(config: dict, class_names: list[str]) -> dict[int, float]:
    """Inverse-frequency weighting from prepare_data.py's saved counts, so
    an imbalanced dataset doesn't bias the model toward the largest class."""
    counts_path = REPO_ROOT / config["paths"]["class_counts_path"]
    with open(counts_path) as f:
        counts = json.load(f)

    train_counts = {c: counts[c]["train"] for c in class_names}
    total = sum(train_counts.values())
    n_classes = len(class_names)

    weights = {}
    for i, name in enumerate(class_names):
        weights[i] = total / (n_classes * train_counts[name])
    return weights


def build_model(config: dict, num_classes: int):
    img_size = tuple(config["image"]["size"])
    aug = config["augmentation"]

    inputs = layers.Input(shape=(None, None, 3))  # accepts any upload size

    x = layers.Resizing(img_size[0], img_size[1])(inputs)

    # Augmentation layers: active only when the model is called with
    # training=True (i.e. during model.fit). Identity at inference/export.
    x = layers.RandomFlip(aug["random_flip"])(x)
    x = layers.RandomRotation(aug["random_rotation"])(x)
    x = layers.RandomZoom(aug["random_zoom"])(x)
    x = layers.RandomContrast(aug["random_contrast"])(x)

    # Replicates tf.keras.applications.mobilenet_v2.preprocess_input
    # (x / 127.5 - 1.0) as a plain, fully-serializable layer.
    x = layers.Rescaling(scale=1 / 127.5, offset=-1.0)(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config["train"]["dropout"])(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="croplens")
    return model, base_model


def compile_model(model, learning_rate: float, label_smoothing: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tf.keras.utils.set_random_seed(config["seed"])

    checkpoint_dir = REPO_ROOT / config["paths"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets(config)
    print(f"Classes (in label order): {class_names}")

    labels_path = REPO_ROOT / config["paths"]["labels_path"]
    with open(labels_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved label order to {labels_path} - export.py will reuse this exact file.")

    class_weight = compute_class_weight(config, class_names)
    print(f"Class weights: {class_weight}")

    model, base_model = build_model(config, num_classes=len(class_names))
    model.summary()

    best_ckpt_path = str(REPO_ROOT / config["paths"]["best_checkpoint"])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config["train"]["early_stopping_patience"],
            restore_best_weights=True,
        ),
    ]

    # ---- Phase 1: frozen base, train the head ----
    print("\n=== Phase 1: training head (base frozen) ===")
    compile_model(
        model, config["train"]["learning_rate_head"], config["train"]["label_smoothing"]
    )
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["train"]["epochs_head"],
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # ---- Phase 2: fine-tune top layers ----
    print("\n=== Phase 2: fine-tuning top layers ===")
    base_model.trainable = True
    fine_tune_at = config["train"]["fine_tune_at_layer"]
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    compile_model(
        model,
        config["train"]["learning_rate_fine_tune"],
        config["train"]["label_smoothing"],
    )
    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["train"]["epochs_fine_tune"],
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Reload the best checkpoint in case fine-tuning ended worse than it started.
    model = tf.keras.models.load_model(best_ckpt_path)

    combined_history = {}
    for h in (history_1.history, history_2.history):
        for k, v in h.items():
            combined_history.setdefault(k, []).extend(v)

    history_path = REPO_ROOT / config["paths"]["history_path"]
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nBest model - val_accuracy: {val_acc:.4f}, val_loss: {val_loss:.4f}")
    print(f"Saved to {best_ckpt_path}")
    print("Next: python training/evaluate.py")


if __name__ == "__main__":
    main()
