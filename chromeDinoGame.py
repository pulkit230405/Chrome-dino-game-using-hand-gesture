import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Variable to keep track of hand closed status in the previous frame
hand_closed_prev = False

# Variables for FPS calculation
prev_time = 0
fps = 0

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Calculate FPS
    current_time = time.time()
    fps += 1
    if current_time - prev_time >= 1:
        fps_text = f"FPS: {fps}"
        prev_time = current_time
        fps = 0
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Detect hand
    # Function to detect hand and draw landmarks
    def detect_hand(frame):
        global hand_closed_prev

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Get landmarks of the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks and connect lines
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)  # Purple color for landmarks
            
            # Convert hand landmarks to numpy array of points
            hand_points = np.array([(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in hand_landmarks.landmark])

            # Draw lines between specific landmarks
            connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                           (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                           (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                           (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                           (0, 17), (17, 18), (18, 19), (19, 20)]  # Little finger
            for connection in connections:
                cv2.line(frame, tuple(hand_points[connection[0]]), tuple(hand_points[connection[1]]), (0, 255, 0), 2)

            # Draw bounding box around the hand
            x, y, w, h = cv2.boundingRect(hand_points)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Get the position of the tip of the index finger (landmark index 8)
            index_tip = hand_landmarks.landmark[8]
            # Get the position of the tip of the thumb (landmark index 4)
            thumb_tip = hand_landmarks.landmark[4]

            # Convert landmarks to pixel coordinates
            h, w, c = frame.shape
            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Calculate the distance between the index finger and thumb
            distance = ((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2) ** 0.5

            # If the distance is less than a threshold, it means the hand is closed
            hand_closed = distance < 50

            # Check if hand is closed in the current frame and wasn't closed in the previous frame
            if hand_closed and not hand_closed_prev:
                return True

            # Update hand_closed_prev
            hand_closed_prev = hand_closed

        return False

    # Detect hand
    hand_closed = detect_hand(frame)
    
    if hand_closed:
        # Press space key to make the Dino jump
        pyautogui.press('space')
        # Display jump status in the top right corner
        jump_status = "Jump"
    else:
        # Display no jump status in the top right corner
        jump_status = "No Jump"
    
    cv2.putText(frame, jump_status, (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display frame
    cv2.imshow('Hand Tracking', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
