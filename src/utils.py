def get_class_names(class_indices_path=None):
	"""Return list of class names sorted by index."""
	import json
	from pathlib import Path
	if class_indices_path is None:
		# Default path relative to src
		base_dir = Path(__file__).resolve().parent.parent
		class_indices_path = base_dir / 'data' / 'splits' / 'class_indices.json'
	with open(class_indices_path, 'r') as f:
		class_to_idx = json.load(f)
	# Sort by index
	idx_to_class = {int(v): k for k, v in class_to_idx.items()}
	return [idx_to_class[i] for i in sorted(idx_to_class.keys())]
import json
from pathlib import Path


def load_class_map(class_indices_path):
	with open(class_indices_path, 'r') as f:
		class_to_idx = json.load(f)
	idx_to_class = {int(v): k for k, v in class_to_idx.items()}
	return class_to_idx, idx_to_class




def majority_vote(labels):
	from collections import Counter
	if not labels:
		return None
	return Counter(labels).most_common(1)[0][0]




def smooth_predictions(preds, window=9):
	"""Simple moving-average smoothing over per-frame class probabilities.
	preds: list of np.array softmax vectors
	returns: list of smoothed np.array
	"""
	import numpy as np
	if len(preds) == 0:
		return []
	k = max(1, window)
	c = preds[0].shape[-1]
	arr = np.stack(preds) # T x C
	smoothed = []
	for i in range(len(preds)):
		a = max(0, i - k//2)
		b = min(len(preds), i + k//2 + 1)
		smoothed.append(arr[a:b].mean(axis=0))
	return smoothed