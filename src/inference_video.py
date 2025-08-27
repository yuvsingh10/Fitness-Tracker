
import cv2
import numpy as np
import tensorflow as tf
from rep_counter import RepetitionCounter
from utils import smooth_predictions, get_class_names
try:
    import mediapipe as mp
except Exception:
    mp = None

# Path to trained model
MODEL_PATH = "models/mobilenetv2_correctness.h5"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
class_names = get_class_names()

# Initialize repetition counter
rep_counter = RepetitionCounter()

def run_inference(video_path, output_path="output.avi"):
    """
    Run inference on a given video, classify exercises,
    count repetitions, and save output with overlays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    preds_window = []
    smoothing_window = 10

    if mp is not None:
        pose = mp.solutions.pose.Pose(static_image_mode=False)
    else:
        pose = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess for MobileNetV2
        img = cv2.resize(frame, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Model prediction
        preds = model.predict(img, verbose=0)
        confidence = np.max(preds)

        # Smooth predictions (use softmax vectors)
        preds_window.append(preds[0])
        if len(preds_window) > smoothing_window:
            preds_window.pop(0)
        smoothed = smooth_predictions(preds_window)
        smooth_class = np.argmax(smoothed[-1])

        # Run pose detection and update rep counter
        if pose is not None:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                rep_counter.update(results.pose_landmarks.landmark)
            else:
                rep_counter.update(frame)
        else:
            rep_counter.update(frame)

        # Overlay info
        cv2.putText(frame, f"{class_names[smooth_class]} ({confidence:.2f})",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(frame, f"Reps: {rep_counter.count}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow("Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processed video saved at: {output_path}")


if __name__ == "__main__":
    # Example usage
    run_inference("data/raw_videos/correct pushup/Correct Push Up 1.mp4")
