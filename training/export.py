"""
Day 3: package the trained model for deployment.

Copies the best checkpoint + its label order + evaluate.py's metrics into
models/v1/, and generates a starter MODEL_CARD.md. Also does a smoke test:
reload the exported model fresh and run one prediction, so a serialization
bug is caught here instead of after deployment.

Run:
    python training/export.py
"""

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = "training/config.yaml") -> dict:
    with open(REPO_ROOT / path) as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def render_model_card(config: dict, metrics: dict, class_names: list[str], model_size_mb: float) -> str:
    per_class_rows = "\n".join(
        f"| {name} | {metrics['per_class'][name]['precision']:.3f} "
        f"| {metrics['per_class'][name]['recall']:.3f} "
        f"| {metrics['per_class'][name]['f1']:.3f} "
        f"| {metrics['per_class'][name]['support']} |"
        for name in class_names
    )

    return f"""# Model card - CropLens {config['paths']['model_version']}

**Exported:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Git commit:** {get_git_commit()}

## Overview

- **Architecture:** MobileNetV2 (ImageNet weights) + custom classification head
- **Classes ({len(class_names)}):** {', '.join(class_names)}
- **Input:** any RGB image (resized to {tuple(config['image']['size'])} internally)
- **Preprocessing:** baked into the model (resize + rescale to [-1, 1]) -
  the API sends raw decoded pixels, nothing extra to keep in sync
- **Model size:** {model_size_mb:.1f} MB

## Test-set results

- **Accuracy:** {metrics['test_accuracy']:.3f}
- **Macro F1:** {metrics['macro_f1']:.3f}

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
{per_class_rows}

Confusion matrix: `docs/results/confusion_matrix.png`

## Confidence threshold

Chosen from `docs/results/threshold_search.json` (validation set).
<!-- TODO: record the value you chose and why -->
Current default in `.env.example`: `{config['train'].get('confidence_threshold', 'see .env.example')}`

## Known limitations

<!-- TODO: fill in honestly, e.g. -->
- Only recognizes the {len(class_names)} classes listed above
- Trained on a limited dataset; accuracy may be lower on unusual lighting,
  heavy occlusion, or non-phone-camera images
- Background-removed training images may make the model slightly more
  sensitive to backgrounds it wasn't trained on

## Reproducing this model

```bash
python training/prepare_data.py
python training/train.py --config training/config.yaml
python training/evaluate.py
python training/export.py
```

Config used: `training/config.yaml` (seed {config['seed']})
"""


def smoke_test(model_path: Path, labels_path: Path):
    """Reload the exported model exactly as the API would, and run one
    prediction, to catch serialization issues before deployment."""
    model = tf.keras.models.load_model(model_path)
    with open(labels_path) as f:
        labels = json.load(f)

    dummy_image = np.random.randint(0, 255, size=(1, 400, 300, 3), dtype=np.uint8)
    probs = model.predict(dummy_image, verbose=0)[0]

    assert len(probs) == len(labels), (
        f"Model outputs {len(probs)} classes but labels.json has {len(labels)}"
    )
    assert abs(probs.sum() - 1.0) < 1e-3, "Output doesn't sum to ~1 - check the final layer"

    top = labels[int(np.argmax(probs))]
    print(f"  Smoke test OK - dummy image -> '{top}' ({probs.max():.3f})")


def main() -> None:
    config = load_config()

    checkpoint_path = REPO_ROOT / config["paths"]["best_checkpoint"]
    labels_src = REPO_ROOT / config["paths"]["labels_path"]
    metrics_src = REPO_ROOT / "training" / "checkpoints" / "metrics.json"

    for required, name in [
        (checkpoint_path, "best checkpoint (run train.py first)"),
        (labels_src, "labels.json (run train.py first)"),
        (metrics_src, "metrics.json (run evaluate.py first)"),
    ]:
        if not required.exists():
            raise FileNotFoundError(f"Missing {name}: {required}")

    with open(labels_src) as f:
        class_names = json.load(f)
    with open(metrics_src) as f:
        metrics = json.load(f)

    model_out = REPO_ROOT / config["paths"]["model_out"]
    labels_out = REPO_ROOT / config["paths"]["labels_out"]
    metrics_out = REPO_ROOT / config["paths"]["metrics_out"]
    card_out = REPO_ROOT / config["paths"]["model_card_out"]
    model_out.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, model_out)
    shutil.copy2(labels_src, labels_out)

    model_size_mb = model_out.stat().st_size / (1024 * 1024)

    metrics["exported_at"] = datetime.now(timezone.utc).isoformat()
    metrics["git_commit"] = get_git_commit()
    metrics["model_version"] = config["paths"]["model_version"]
    metrics["model_size_mb"] = round(model_size_mb, 2)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    card_out.write_text(render_model_card(config, metrics, class_names, model_size_mb))

    print(f"Exported model  -> {model_out} ({model_size_mb:.1f} MB)")
    print(f"Exported labels -> {labels_out}")
    print(f"Exported metrics -> {metrics_out}")
    print(f"Model card      -> {card_out}")

    print("\nRunning smoke test on the exported model...")
    smoke_test(model_out, labels_out)

    print(
        f"\nDone. models/{config['paths']['model_version']}/ is ready to commit "
        "and use in the API."
    )


if __name__ == "__main__":
    main()
