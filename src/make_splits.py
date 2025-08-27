import json
from pathlib import Path
import random


BASE_DIR = Path(__file__).resolve().parent.parent
FRAMES_DIR = BASE_DIR / 'data' / 'frames'
SPLIT_DIR = BASE_DIR / 'data' / 'splits'
VAL_RATIO = 0.2


random.seed(42)


SPLIT_DIR.mkdir(parents=True, exist_ok=True)


classes = sorted([d.name for d in FRAMES_DIR.iterdir() if d.is_dir()])
class_to_idx = {c: i for i, c in enumerate(classes)}
with open(SPLIT_DIR/'class_indices.json', 'w') as f:
	json.dump(class_to_idx, f)


all_paths = []
for c in classes:
	for vid_dir in (FRAMES_DIR/c).iterdir():
		if not vid_dir.is_dir():
			continue
		frames = sorted(list(vid_dir.glob('*.jpg')))
		if len(frames) == 0:
			continue
		# Keep frames grouped by video so val doesn't see same video
		all_paths.append((c, [str(p.resolve()) for p in frames]))


random.shuffle(all_paths)
val_count = int(len(all_paths) * VAL_RATIO)
val_items = all_paths[:val_count]
train_items = all_paths[val_count:]


with open(SPLIT_DIR/'train.txt', 'w') as f:
	for c, frames in train_items:
		for fp in frames:
			f.write(f"{fp} {class_to_idx[c]}\n")


with open(SPLIT_DIR/'val.txt', 'w') as f:
	for c, frames in val_items:
		for fp in frames:
			f.write(f"{fp} {class_to_idx[c]}\n")


print('Wrote splits and class indices.')