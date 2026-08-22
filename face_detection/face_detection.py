import cv2
import mediapipe as mp
import os
import urllib.request
import traceback


# =====================================================
# 1. Model
# =====================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/latest/"
    "blaze_face_short_range.tflite"
)

MODEL_PATH = "blaze_face_short_range.tflite"


# =====================================================
# 2. Download model only once
# =====================================================

if not os.path.exists(MODEL_PATH):

    print("Downloading model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Model downloaded.")


# =====================================================
# 3. MediaPipe
# =====================================================

BaseOptions = mp.tasks.BaseOptions

FaceDetector = mp.tasks.vision.FaceDetector

FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

RunningMode = mp.tasks.vision.RunningMode


# =====================================================
# 4. Detector options
# =====================================================

options = FaceDetectorOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    min_detection_confidence=0.5,

    min_suppression_threshold=0.3
)


# =====================================================
# 5. Main program
# =====================================================

try:

    print("Creating Face Detector...")

    detector = FaceDetector.create_from_options(options)

    print("Face Detector ready.")


    # =================================================
    # Camera
    # =================================================

    print("Opening camera...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Camera open nahi hua.")

        detector.close()

        input("Press Enter to exit...")

        exit()


    print("Camera opened successfully.")
    print("Press Q to quit.")


    # =================================================
    # Timestamp
    # =================================================

    timestamp_ms = 0


    # =================================================
    # Camera loop
    # =================================================

    while True:

        success, frame = cap.read()


        if not success:

            print("ERROR: Camera frame read nahi hua.")

            break


        # =================================================
        # Mirror
        # =================================================

        frame = cv2.flip(frame, 1)


        # =================================================
        # BGR -> RGB
        # =================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # =================================================
        # MediaPipe image
        # =================================================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # =================================================
        # Increasing Timestamp
        # =================================================

        timestamp_ms += 33


        # =================================================
        # Face detection
        # =================================================

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # =================================================
        # Draw faces
        # =================================================

        face_count = len(result.detections)


        for detection in result.detections:

            bbox = detection.bounding_box


            # =================================================
            # Bounding Box
            # =================================================

            x = max(
                0,
                bbox.origin_x
            )

            y = max(
                0,
                bbox.origin_y
            )


            x2 = min(
                frame.shape[1],
                x + bbox.width
            )

            y2 = min(
                frame.shape[0],
                y + bbox.height
            )


            # =================================================
            # Face Box
            # =================================================

            cv2.rectangle(
                frame,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # Confidence
            # =================================================

            if detection.categories:

                confidence = (
                    detection.categories[0].score
                )


                cv2.putText(
                    frame,

                    f"Confidence: "
                    f"{confidence * 100:.1f}%",

                    (
                        x,
                        max(30, y - 10)
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 0),

                    2
                )


            # =================================================
            # Keypoints
            # =================================================

            for keypoint in detection.keypoints:

                px = int(
                    keypoint.x *
                    frame.shape[1]
                )

                py = int(
                    keypoint.y *
                    frame.shape[0]
                )


                cv2.circle(
                    frame,

                    (px, py),

                    4,

                    (0, 0, 255),

                    -1
                )


        # =================================================
        # Face Count
        # =================================================

        cv2.putText(
            frame,

            f"Faces Detected: {face_count}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (255, 255, 255),

            2
        )


        # =================================================
        # Timestamp Display
        # =================================================

        cv2.putText(
            frame,

            f"Timestamp: {timestamp_ms} ms",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        # =================================================
        # Show Camera
        # =================================================

        cv2.imshow(
            "MediaPipe Face Detection",
            frame
        )


        # =================================================
        # Q = Quit
        # =================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            print("Q pressed. Closing...")

            break


# =====================================================
# Error Handling
# =====================================================

except Exception:

    print(
        "\n===================================="
    )

    print(
        "PROGRAM ERROR"
    )

    print(
        "====================================\n"
    )


    traceback.print_exc()


    print(
        "\n===================================="
    )

    input(
        "Press Enter to close..."
    )


# =====================================================
# Cleanup
# =====================================================

finally:

    try:

        cap.release()

    except:

        pass


    try:

        detector.close()

    except:

        pass


    cv2.destroyAllWindows()


    print(
        "Program closed."
    )