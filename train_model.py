import csv
import pickle

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

X = []
y = []

with open("dataset.csv", "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        # Get the letter
        label = row["label"]

        # Get the 63 hand landmark values
        features = []

        for i in range(21):
            features.append(float(row[f"x{i}"]))
            features.append(float(row[f"y{i}"]))
            features.append(float(row[f"z{i}"]))

        X.append(features)
        y.append(label)


print("================================")
print("SIGN LANGUAGE MODEL TRAINING")
print("================================")

print()
print("Total samples:", len(X))

print()
print("Samples per letter:")
print(Counter(y))


# ==========================================
# 2. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 3. CREATE RANDOM FOREST
# ==========================================

print()
print("Training Random Forest...")
print("Please wait...")
print()

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 4. TRAIN
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 5. TEST
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("================================")
print("MODEL RESULTS")
print("================================")

print()
print(f"Accuracy: {accuracy * 100:.2f}%")

print()
print("Classification Report:")
print()

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 6. SAVE MODEL
# ==========================================

with open("sign_language_model.pkl", "wb") as file:

    pickle.dump(model, file)


print()
print("================================")
print("MODEL SAVED SUCCESSFULLY!")
print("================================")
print()
print("File created:")
print("sign_language_model.pkl")