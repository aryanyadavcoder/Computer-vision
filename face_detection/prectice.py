import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

def start_camera():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot access camera")
        return None
    return cap

def Run_camera():
    cap = start_camera()
    if cap is None:
        return
        
    while True:
        success, frame = cap.read()
        if not success:
            print("Frame not avlible")
            break
        cv2.imshow("Camera",frame)
        if cv2.waitKey(1) &0xff == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    Run_camera()    
