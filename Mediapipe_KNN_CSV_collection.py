import cv2
import mediapipe as mp
import csv
import os

# --- CONFIG ---
model_path = "/Users/mayankkumar/Desktop/python 12.46.49 PM/hand_landmarker.task"
csv_file = "drone_gestures_data.csv"
GESTURE_LIST = ["up", "down", "fist", "fly", "dive", "backflip", "ok_home","right","left","spin 360 degree","Audio Recording"]
current_idx = 0 

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

print(f"\n🚀 BOX-MODE DATA COLLECTION")
print("S: Save Data | N: Next Gesture | Q: Quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_count += 1
    
    # 1. Box Define Karo
    box_size = 350
    x1, y1 = (w // 2) - (box_size // 2), (h // 2) - (box_size // 2)
    x2, y2 = x1 + box_size, y1 + box_size
    roi = frame[y1:y2, x1:x2]
    
    # 2. Key Check (Loop ke bahar taaki 'key' hamesha defined rahe)
    key = cv2.waitKey(1) & 0xFF
    
    # 3. MediaPipe ROI par
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    results = detector.detect_for_video(mp_image, frame_count)

    # UI Design
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"TARGET: {GESTURE_LIST[current_idx].upper()}", (x1, y1-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw landmarks
            for lm in hand_landmarks:
                cx, cy = int(lm.x * box_size) + x1, int(lm.y * box_size) + y1
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

            # SAVE Logic (Press 'S')
            if key == ord('s'): 
                wrist_x, wrist_y, wrist_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
                row = []
                for pt in hand_landmarks:
                    row.extend([pt.x - wrist_x, pt.y - wrist_y, pt.z - wrist_z])
                row.append(GESTURE_LIST[current_idx])
                
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"✅ Saved Sample for {GESTURE_LIST[current_idx]}")

    # NEXT Logic (Press 'N') - Ab hand ho ya na ho, ye kaam karega
    if key == ord('n'):
        current_idx = (current_idx + 1) % len(GESTURE_LIST)
        print(f"➡️ Switched to: {GESTURE_LIST[current_idx].upper()}")

    elif key == ord('q'):
        break

    cv2.imshow("Box Data Collector", frame)

detector.close()
cap.release()
cv2.destroyAllWindows()