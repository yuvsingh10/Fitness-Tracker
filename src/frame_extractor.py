import cv2
import os
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
IN_DIR = BASE_DIR / 'data' / 'raw_videos'
OUT_DIR = BASE_DIR / 'data' / 'frames'
TARGET_FPS = 2 # sample 2 frames/sec (tune as needed)
IMG_SIZE = (224, 224)


# Classes are directories under raw_videos


def extract_frames():
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	classes = [d.name for d in IN_DIR.iterdir() if d.is_dir()]
	for cls in classes:
		cls_dir = IN_DIR / cls
		videos = list(cls_dir.glob('*.mp4')) + list(cls_dir.glob('*.mov')) + list(cls_dir.glob('*.avi'))
		for vpath in tqdm(videos, desc=f"{cls}"):
			cap = cv2.VideoCapture(str(vpath))
			if not cap.isOpened():
				print(f"Cannot open {vpath}")
				continue
			fps = cap.get(cv2.CAP_PROP_FPS) or 30
			stride = max(1, int(round(fps / TARGET_FPS)))
			out_subdir = OUT_DIR / cls / vpath.stem
			out_subdir.mkdir(parents=True, exist_ok=True)
			i = 0
			frame_id = 0
			while True:
				ret, frame = cap.read()
				if not ret:
					break
				if i % stride == 0:
					frame = cv2.resize(frame, IMG_SIZE)
					out_path = out_subdir / f"frame_{frame_id:04d}.jpg"
					cv2.imwrite(str(out_path), frame)
					frame_id += 1
				i += 1
			cap.release()


if __name__ == '__main__':
	extract_frames()