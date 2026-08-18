import cv2
import csv
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# MediaPipe setup
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# Create dataset file
# -----------------------------

dataset_file = "dataset.csv"

file_exists = os.path.exists(dataset_file)

with open(dataset_file, "a", newline="") as file:

    writer = csv.writer(file)

    # Create column names if file is new
    if not file_exists:

        header = ["label"]

        for i in range(21):
            header.append(f"x{i}")
            header.append(f"y{i}")
            header.append(f"z{i}")

        writer.writerow(header)


# -----------------------------
# Choose the letter
# -----------------------------

letter = input("Enter the sign/letter you want to collect (A, B, C...): ").upper()

print()
print(f"Collecting data for: {letter}")
print("Show the sign to the camera.")
print("Press SPACE to save a sample.")
print("Press Q to quit.")
print()


# -----------------------------
# Open camera
# -----------------------------

cap = cv2.VideoCapture(0)

sample_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera.")
        break

    frame = cv2.flip(frame, 1)

    # Convert image for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    results = detector.detect(mp_image)

    hand_detected = False
    landmarks = None

    if results.hand_landmarks:

        hand_detected = True
        landmarks = results.hand_landmarks[0]

        h, w, _ = frame.shape

        # Draw hand points
        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    # -----------------------------
    # Display information
    # -----------------------------

    cv2.putText(
        frame,
        f"Sign: {letter}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples saved: {sample_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if hand_detected:

        cv2.putText(
            frame,
            "HAND DETECTED - Press SPACE",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Sign Language Data Collection", frame)


    # -----------------------------
    # Keyboard controls
    # -----------------------------

    key = cv2.waitKey(1) & 0xFF

    # SPACE = save sample
    if key == ord(" "):

        if hand_detected:

            row = [letter]

            for landmark in landmarks:

                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)

            with open(dataset_file, "a", newline="") as file:

                writer = csv.writer(file)
                writer.writerow(row)

            sample_count += 1

            print(f"{letter} saved! Sample {sample_count}")

        else:

            print("No hand detected. Try again.")


    # Q = quit
    elif key == ord("q"):

        break


cap.release()
cv2.destroyAllWindows()
detector.close()

print()
print(f"Finished! Saved {sample_count} samples for {letter}.")