import os
import cv2
import json
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define folder for saving the collected data
DATA_PATH = 'dataset'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Set up webcam feed with higher resolution (increase accuracy)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Custom input for label
sign_label = input("Enter the label: ")

# Create a folder for this sign if it doesn't exist
sign_path = os.path.join(DATA_PATH, sign_label)
if not os.path.exists(sign_path):
    os.makedirs(sign_path)

# Flag to control saving state
is_saving = False
data_buffer = []  # Buffer to store frames before saving

# Start collecting data
with mp_hands.Hands(
    min_detection_confidence=0.75,  # Increased detection confidence
    min_tracking_confidence=0.75    # Increased tracking confidence
) as hands:
    count = 0
    print(f"Collecting data for '{sign_label}'. Press 's' to start/stop recording, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Collect landmark points
                landmarks = []
                for lm in hand_landmarks.landmark:
                    # Normalize the landmark points to make them invariant to image size
                    landmarks.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                    })

                # Data augmentation: flip landmarks horizontally for symmetrical gestures
                flipped_landmarks = [{'x': 1 - lm['x'], 'y': lm['y'], 'z': lm['z']} for lm in landmarks]

                # Store the current frame data in the buffer if saving is active
                if is_saving:
                    data_buffer.append({
                        'original': landmarks,
                        'flipped': flipped_landmarks
                    })
                    print(f"Collecting data... (Buffer size: {len(data_buffer)})")
                                                                                    
        # Show frame
        cv2.imshow('Hand Tracking', frame)

        # Key press handling
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            is_saving = not is_saving  # Toggle saving mode
            if is_saving:
                print("Started saving data.")
            else:
                print("Paused saving data. Saving collected data to files...")
                # Save all data in buffer when saving is toggled off
                for i, data in enumerate(data_buffer, start=count + 1):
                    # Save original landmarks
                    file_name = os.path.join(sign_path, f"{sign_label}_{i}.json")
                    with open(file_name, 'w') as f:
                        json.dump(data['original'], f)
                    print(f"Saved {file_name}")

                    # Save flipped landmarks (augmentation)
                    flipped_file_name = os.path.join(sign_path, f"{sign_label}_flipped_{i}.json")
                    with open(flipped_file_name, 'w') as f:
                        json.dump(data['flipped'], f)
                    print(f"Saved {flipped_file_name} (flipped)")
                count += len(data_buffer)
                data_buffer.clear()  # Clear buffer after saving
        elif key == ord('q'):
            print("Exiting...")
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
