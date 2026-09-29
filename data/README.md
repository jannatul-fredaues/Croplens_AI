# Results

Generated automatically - don't hand-write these, just run the pipeline:

```bash
python training/evaluate.py --gradcam
```

This produces:
- `confusion_matrix.png` - test-set confusion matrix
- `threshold_search.json` - coverage vs. accuracy at each candidate
  confidence threshold (validation set) - see `evaluate.py`'s
  `threshold_search()` for how to read it
- `gradcam/<class>_<n>.png` - Grad-CAM heatmap examples per class
  (omit `--gradcam` to skip these; they're slower to generate)

Re-run any time you retrain, so these stay in sync with `models/v1/`.
