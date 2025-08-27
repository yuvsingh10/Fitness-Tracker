import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from inference_video import run_inference


class FitnessApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Fitness Tracker - Posture & Rep Counter")
		self.root.geometry("500x400")
		self.root.configure(bg="#f0f4f7")

		title = tk.Label(self.root, text="Fitness Tracker", font=("Arial", 20, "bold"), bg="#f0f4f7", fg="#2d6a4f")
		title.pack(pady=(20, 10))

		instructions = tk.Label(
			self.root,
			text="Upload a pushup video to analyze posture and count reps.",
			font=("Arial", 12),
			bg="#f0f4f7",
			fg="#1b4332"
		)
		instructions.pack(pady=(0, 20))

		self.upload_btn = tk.Button(
			self.root,
			text="Upload Video",
			command=self.upload_video,
			font=("Arial", 14, "bold"),
			bg="#40916c",
			fg="white",
			activebackground="#52b788",
			activeforeground="white",
			relief=tk.RAISED,
			bd=3,
			padx=10,
			pady=5
		)
		self.upload_btn.pack(pady=20)

		self.result_label = tk.Label(
			self.root,
			text="",
			font=("Arial", 14),
			bg="#f0f4f7",
			fg="#081c15",
			wraplength=400,
			justify="center"
		)
		self.result_label.pack(pady=20)

	def upload_video(self):
		file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
		if not file_path:
			return
		messagebox.showinfo("Processing", "Video processing started...")

		# Run inference in a thread to avoid freezing UI
		# Run inference in a thread to avoid freezing UI, but do not save output video
		def gui_inference():
			import numpy as np
			import cv2
			from inference_video import model, class_names, rep_counter, smooth_predictions
			cap = cv2.VideoCapture(file_path)
			preds_window = []
			smoothing_window = 10
			try:
				import mediapipe as mp
				pose = mp.solutions.pose.Pose(static_image_mode=False)
			except Exception:
				pose = None
			# Get screen size and set window to half
			import ctypes
			user32 = ctypes.windll.user32
			screen_width = user32.GetSystemMetrics(0)
			screen_height = user32.GetSystemMetrics(1)
			win_width = int(screen_width / 2)
			win_height = int(screen_height / 2)
			cv2.namedWindow("Posture & Rep Count", cv2.WINDOW_NORMAL)
			cv2.resizeWindow("Posture & Rep Count", win_width, win_height)
			while True:
				ret, frame = cap.read()
				if not ret:
					break
				img = cv2.resize(frame, (224, 224))
				img = img.astype("float32") / 255.0
				img = np.expand_dims(img, axis=0)
				preds = model.predict(img, verbose=0)
				confidence = np.max(preds)
				preds_window.append(preds[0])
				if len(preds_window) > smoothing_window:
					preds_window.pop(0)
				smoothed = smooth_predictions(preds_window)
				smooth_class = np.argmax(smoothed[-1])
				if pose is not None:
					results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
					if results.pose_landmarks:
						rep_counter.update(results.pose_landmarks.landmark)
					else:
						rep_counter.update(frame)
				else:
					rep_counter.update(frame)
				cv2.putText(frame, f"{class_names[smooth_class]} ({confidence:.2f})",
							(30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
							(0, 255, 0), 2)
				cv2.imshow("Posture & Rep Count", frame)
				# Stop processing if window is closed
				if cv2.getWindowProperty("Posture & Rep Count", cv2.WND_PROP_VISIBLE) < 1:
					break
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
			cap.release()
			cv2.destroyAllWindows()
		thread = threading.Thread(target=gui_inference)
		thread.start()


if __name__ == "__main__":
	root = tk.Tk()
	app = FitnessApp(root)
	root.mainloop()