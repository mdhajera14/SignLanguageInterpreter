import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the hand landmark model
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not access camera")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    results = detector.detect(mp_image)

    # Draw landmarks
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:

            h, w, _ = frame.shape

            # Draw points
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:

                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]

                start_point = (
                    int(start.x * w),
                    int(start.y * h)
                )

                end_point = (
                    int(end.x * w),
                    int(end.y * h)
                )

                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    (0, 255, 0),
                    2
                )

    # Show camera
    cv2.imshow("Sign Language - Hand Tracking", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()