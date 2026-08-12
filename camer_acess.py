import cv2
import time


# 1. Camera start
def start_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access camera")
        return None
    return cap

# 2. Photo save
def save_photo(frame, img_counter):
    img_name = f"classroom_picture_{img_counter}.png"
    cv2.imwrite(img_name, frame)
    print(f"Saved: {img_name}")
    return img_name

# 3. Show saved photo
def show_photo(img_name):
    saved_image = cv2.imread(img_name)

    if saved_image is None:
        print("Photo not found")
        return

    cv2.imshow("Saved Picture", saved_image)
    print("Saved photo")
    cv2.waitKey(2000)
    cv2.destroyWindow("Saved Picture")

# 4. Camera release
def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()


# 5. Camera function
def run_camera():

    cap = start_camera()

    if cap is None:
        return None

    print("Camera opened")

    start_time = time.time()
    frame = None
    while time.time() - start_time < 5:

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("AI Smart Classroom - Camera", frame)
        cv2.waitKey(1)

    if frame is not None:
        print("Taking photo")
        saved_image = save_photo(frame, 0)
    else:
        saved_image = None
    # Camera close
    close_camera(cap)
    print("Camera closed")
    return saved_image

# 6. Main program
def main():
    print("\n---- Menu----")

    print("Commands:")
    print("open camera")
    print("save photo")
    print("show photo")
    print("quit")
    user_input = input("\nEnter commands(,): ")
    commands = user_input.split(",")
    saved_image = None
    for command in commands:
        command = command.strip().lower()
        if command == "open camera":
            print("\n Opening Camera...")
            saved_image = run_camera()
        elif command == "save photo":
            print("\n Save Photo command received.")

            if saved_image is not None:
                print("Photo already saved:", saved_image)
            else:
                print("No camera photo available.")

        elif command == "show photo":

            print("\n Showing Photo")

            if saved_image is not None:
                show_photo(saved_image)
            else:
                print("No photo available.")

        elif command == "quit":

            print("\n Program closed.")
            break

        else:
            print("Unknown command:", command)

# 7. Program start
if __name__ == "__main__":
    main()