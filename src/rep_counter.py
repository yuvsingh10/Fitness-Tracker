
import math
from dataclasses import dataclass
import numpy as np

try:
	import mediapipe as mp
except Exception:
	mp = None

@dataclass
class RepResult:
	count: int
	phase: str
	last_angle: float
	feedback: str

def angle(a, b, c):
	"""Return angle ABC (in degrees). Points are (x, y)."""
	ax, ay = a
	bx, by = b
	cx, cy = c
	ab = (ax - bx, ay - by)
	cb = (cx - bx, cy - by)
	dot = ab[0]*cb[0] + ab[1]*cb[1]
	mag_ab = (ab[0]**2 + ab[1]**2)**0.5
	mag_cb = (cb[0]**2 + cb[1]**2)**0.5
	if mag_ab * mag_cb == 0:
		return 0.0
	cosang = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
	return math.degrees(math.acos(cosang))

class RepCounter:
	"""Counts reps when angle crosses thresholds up & down."""
	def __init__(self, kind='squat'):
		self.kind = kind
		self.state = 'up'
		self.count = 0
		self.last_angle = 180
		self.feedback = ''

	def update(self, input_data):
		"""
		Accepts either pose landmarks (list of objects with .x/.y) or a frame (np.ndarray).
		Only counts reps if landmarks are available.
		"""
		# If input is a numpy array (frame), skip rep counting
		if isinstance(input_data, np.ndarray):
			return
		# If input is a list and has .x/.y attributes, do rep counting
		if self.kind == 'squat' and hasattr(input_data, '__getitem__') and hasattr(input_data[24], 'x'):
			hip = (input_data[24].x, input_data[24].y)
			knee = (input_data[26].x, input_data[26].y)
			ankle = (input_data[28].x, input_data[28].y)
			ang = angle(hip, knee, ankle)
			# Add your rep counting logic here

# Alias for compatibility
RepetitionCounter = RepCounter