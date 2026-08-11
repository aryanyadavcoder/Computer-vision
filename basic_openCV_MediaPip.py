
import cv2
import mediapipe as mp

# Camera start
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mphand = mp.solutions.hands
hands = mphand.Hands()

# Drawing utility
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    if not success:
        print("Camera frame nahi mil raha")
        break

    # BGR -> RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hand detection
    result = hands.process(imgRGB)

    # Agar hand mila
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:

            # Hand landmarks draw karo
            mpDraw.draw_landmarks(
                img,
                handlms,
                mphand.HAND_CONNECTIONS
            )

    # Show camera
    cv2.imshow("Image", img)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

