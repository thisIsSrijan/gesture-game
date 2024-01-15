import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

print("Hello")
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')

with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Remove any empty strings from classNames
classNames = [name for name in classNames if name]

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Extracting relevant landmarks for clenched fist with right-pointing thumb
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            # Check if thumb is pointing towards the right and other fingers are close to thumb (clenched fist)
            thumb_is_right = (
                thumb_tip[0] > index_tip[0]
                and thumb_tip[0] > middle_tip[0]
                and thumb_tip[0] > ring_tip[0]
                and thumb_tip[0] > pinky_tip[0]
            )

            # Calculate the distance between thumb tip and index finger tip
            distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)

            # Apply additional filtering for improved precision
            if (
                thumb_is_right
                and distance > 45
                # and thumb_tip[1] < index_tip[1]
                # and thumb_tip[1] < middle_tip[1]
                # and thumb_tip[1] < ring_tip[1]
                # and thumb_tip[1] < pinky_tip[1]
            ):
                className = "Right Pointing Thumb with Clenched Fist"
            else:
                className = "Unknown"

            hand_side = "Right" if landmarks[0][0] > x / 2 else "Left"

            text_position = (10, 50) if hand_side == "Right" else (x - 350, 50)

            cv2.putText(
                frame,
                f"{hand_side} Hand: {className}",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
