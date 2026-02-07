import cv2
import mediapipe as mp
import joblib
import numpy as np

# --- CONFIG ---
model_path = "/Users/mayankkumar/Desktop/python 12.46.49 PM/hand_landmarker.task"
knn_model_path = "/Users/mayankkumar/Desktop/python 12.46.49 PM/gesture_model_3.pkl"

# Load the trained KNN model
model = joblib.load(knn_model_path)

# --- MEDIAPIPE SETUP ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
frame_count = 0

print("🔥 LIVE PREDICTION STARTED")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_count += 1
    
    # 1. Box ROI (Same as collection mode)
    box_size = 350
    x1, y1 = (w // 2) - (box_size // 2), (h // 2) - (box_size // 2)
    x2, y2 = x1 + box_size, y1 + box_size
    roi = frame[y1:y2, x1:x2]
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    results = detector.detect_for_video(mp_image, frame_count)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue box

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # A. Landmarks ko feature vector mein convert karo
            wrist_x, wrist_y, wrist_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
            features = []
            for pt in hand_landmarks:
                features.extend([pt.x - wrist_x, pt.y - wrist_y, pt.z - wrist_z])
            
            # B. Prediction & Confidence
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])
            confidence = np.max(probability) * 100

            # C. Color Logic based on Confidence
            color = (0, 255, 0) if confidence > 80 else (0, 0, 255)
            
            # D. Display Result
            cv2.putText(frame, f"CMD: {prediction.upper()}", (x1, y1-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(frame, f"CONF: {confidence:.1f}%", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # E. Draw skeleton (Fixed the 'lm' error here)
            for pt in hand_landmarks:
                cx, cy = int(pt.x * box_size) + x1, int(pt.y * box_size) + y1
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    cv2.imshow("Drone AI Controller - Test Mode", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

detector.close()
cap.release()
cv2.destroyAllWindows()