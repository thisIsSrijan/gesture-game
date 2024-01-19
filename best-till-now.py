import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

print("Hello")
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')



def calculateDistance(point1,point2):
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance


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

            wrist_end = landmarks[0]
            thumb_end = landmarks[2]
            index_end = landmarks[5]
            middle_end = landmarks[9]
            ring_end = landmarks[13]
            pinky_end = landmarks[17]

            # Check if thumb is pointing towards the right and other fingers are close to thumb (clenched fist)
            palm_shown = (
                thumb_tip[1] < wrist_end[1]
                and index_tip[1] < wrist_end[1]
                and middle_tip[1] < wrist_end[1]
                and ring_tip[1] < wrist_end[1]
                and pinky_tip[1] < wrist_end[1]
            )

            chench_fist = (
                thumb_tip[1] < wrist_end[1]
                and index_tip[1] < wrist_end[1]
                and middle_tip[1] < wrist_end[1]
                and ring_tip[1] < wrist_end[1]
                and pinky_tip[1] < wrist_end[1]
            )

            thumb_is_right = (
                thumb_tip[0] > index_tip[0]
                and thumb_tip[0] > middle_tip[0]
                and thumb_tip[0] > ring_tip[0]
                and thumb_tip[0] > pinky_tip[0]
            )


            thumb_is_left = (
                thumb_tip[0] < index_tip[0]
                and thumb_tip[0] < middle_tip[0]
                and thumb_tip[0] < ring_tip[0]
                and thumb_tip[0] < pinky_tip[0]
            )

            # Calculate the distance between points
            tip_to_thumb_distance = calculateDistance(thumb_tip, index_tip)
            index_tip_to_end = calculateDistance(index_tip, index_end)
            middle_tip_to_end = calculateDistance(middle_end, middle_end)
            ring_tip_to_end = calculateDistance(ring_tip, ring_end)
            pinky_tip_to_end = calculateDistance(pinky_tip, pinky_end)
            thumb_tip_to_end = calculateDistance(thumb_tip, thumb_end)


            wrist_to_middle = calculateDistance(middle_tip, wrist_end)


            gesture_map = {
                "move_left" : False, "move_right" : False , "hold" : False, "attack" : False
            }

            # Apply additional filtering for improved precision
            if (
                thumb_is_right
                and tip_to_thumb_distance > 40
                and tip_to_thumb_distance < 90
                
            ):
                className = "Right Pointing Thumb with Clenched Fist"
                gesture_map = {key: False for key in gesture_map}
                gesture_map["move_right"] = True
            
            elif (
                thumb_is_left
                and tip_to_thumb_distance > 40
                and tip_to_thumb_distance < 90
                
            ):
                className = "Left Pointing Thumb with Clenched Fist"
                gesture_map = {key: False for key in gesture_map}
                gesture_map["move_left"] = True

            elif (
                chench_fist
                and index_tip_to_end < 70
                and middle_tip_to_end < 70
                and ring_tip_to_end < 70
                and pinky_tip_to_end < 70
                and thumb_tip_to_end < 70

            ):
                className = "Fist"
                gesture_map = {key: False for key in gesture_map}
                gesture_map["attack"] = True

            elif (
                palm_shown
                and wrist_to_middle > 60
                # and index_tip_to_end > 20
                # and middle_tip_to_end > 20
                # and ring_tip_to_end > 20
                # and pinky_tip_to_end > 20
            ):
                className = "Palm Shown"
                gesture_map = {key: False for key in gesture_map}
                gesture_map["hold"] = True
            
            else:
                className = "Unknown"

            hand_side = "Right" if landmarks[0][0] > x / 2 else "Left"

            text_position = (10, 50) if hand_side == "Right" else (x - 350, 50)

            cv2.putText(
                frame,
                f"{hand_side} Hand: attack: {gesture_map['attack']}",
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{hand_side} Hand: hold: {gesture_map['hold']}",
                (10,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{hand_side} Hand: move_left: {gesture_map['move_left']}",
                (10,110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{hand_side} Hand: move_right: {gesture_map['move_right']}",
                (10,140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
