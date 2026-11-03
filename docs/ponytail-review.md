backend/src/sentinel/models/__init__.py:L3: delete: unused CNN package re-exports eagerly load PyTorch for every models submodule. Import sentinel.models.cnn directly where needed; remove lines 3–5.

backend/src/sentinel/training/metrics.py:L73: shrink: one-caller _confusion_counts wrapper and integer conversions. Inline confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel() at L57; delete the helper and its two leading blank lines.

net: -12 lines possible.
