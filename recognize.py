import cv2
import pickle
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. LOAD TRAINED RANDOM FOREST MODEL
# ==========================================

with open("sign_language_model.pkl", "rb") as file:
    model = pickle.load(file)

print("Sign language model loaded successfully!")


# ==========================================
# 2. SET UP MEDIAPIPE
# ==========================================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# ==========================================
# 3. OPEN CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started!")
print("Show a sign to the camera.")
print("Press Q to quit.")


# ==========================================
# 4. RECOGNITION LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera.")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    results = detector.detect(mp_image)


    # ==========================================
    # 5. IF HAND IS DETECTED
    # ==========================================

    if results.hand_landmarks:

        landmarks = results.hand_landmarks[0]

        h, w, _ = frame.shape

        # Draw landmarks
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


        # ==========================================
        # 6. CREATE 63 FEATURES
        # ==========================================

        features = []

        for landmark in landmarks:

            features.append(landmark.x)
            features.append(landmark.y)
            features.append(landmark.z)


        # ==========================================
        # 7. PREDICT LETTER
        # ==========================================

        prediction = model.predict([features])

        letter = prediction[0]


        # ==========================================
        # 8. GET CONFIDENCE
        # ==========================================

        probabilities = model.predict_proba([features])

        confidence = max(probabilities[0]) * 100


        # ==========================================
        # 9. DISPLAY PREDICTION
        # ==========================================

        cv2.rectangle(
            frame,
            (10, 10),
            (400, 100),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Sign: {letter}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    else:

        # No hand detected

        cv2.putText(
            frame,
            "Show your hand",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ==========================================
    # 10. SHOW CAMERA
    # ==========================================

    cv2.imshow(
        "Sign Language Recognition",
        frame
    )


    # ==========================================
    # 11. QUIT
    # ==========================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ==========================================
# 12. CLEAN UP
# ==========================================

cap.release()

cv2.destroyAllWindows()

detector.close()

print("Program closed.")