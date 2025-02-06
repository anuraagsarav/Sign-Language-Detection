import cv2
import pickle
import warnings
import numpy as np
import mediapipe as mp

warnings.filterwarnings('ignore', category=UserWarning)

# Load the trained model
with open('models/sign_language_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Mediapipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the webcam feed and increase resolution
cap = cv2.VideoCapture(0)   

# Set the desired resolution (1280x720 or 1920x1080 for higher quality)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Predict the sign using the trained model
                prediction = model.predict([landmarks])
                print(f"Predicted Sign: {prediction[0]}")

                # Display the predicted sign on the frame
                cv2.putText(frame, prediction[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-time Sign Prediction', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
