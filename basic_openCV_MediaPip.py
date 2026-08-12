import cv2
import mediapipe as mp

# Start the camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mphand = mp.solutions.hands
hands = mphand.Hands()

# Drawing utility
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    if not success:
        print("Unable to read the camera frame.")
        break

    # Convert BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(imgRGB)

    # Check for detected hands
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:

            # Draw hand landmarks
            mpDraw.draw_landmarks(
                img,
                handlms,
                mphand.HAND_CONNECTIONS
            )

    # Display the camera
    cv2.imshow("Hand Detection", img)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Close all windows
cv2.destroyAllWindows()