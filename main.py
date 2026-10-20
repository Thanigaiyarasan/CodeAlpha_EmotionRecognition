# CodeAlpha - Speech Emotion Recognition

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Dataset path
DATASET_PATH = "Audio_Speech_Actors_01-24"

# RAVDESS emotion labels
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def extract_features(file_path):
    """Extract audio features from a WAV file."""
    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0)


def get_emotion(file_name):
    """Read emotion label from the RAVDESS filename."""
    parts = file_name.split("-")
    emotion_code = parts[2]
    return EMOTIONS.get(emotion_code)


def main():
    features = []
    labels = []

    print("Loading RAVDESS audio dataset...")

    for actor_folder in sorted(os.listdir(DATASET_PATH)):
        actor_path = os.path.join(DATASET_PATH, actor_folder)

        if not os.path.isdir(actor_path):
            continue

        for file_name in sorted(os.listdir(actor_path)):
            if not file_name.endswith(".wav"):
                continue

            emotion = get_emotion(file_name)

            if emotion is None:
                continue

            file_path = os.path.join(actor_path, file_name)

            try:
                audio_features = extract_features(file_path)
                features.append(audio_features)
                labels.append(emotion)

            except Exception as error:
                print(f"Skipping {file_name}: {error}")

    if not features:
        print("No audio features extracted. Check the dataset path.")
        return

    X = np.array(features)
    y = np.array(labels)

    print("Total audio samples:", len(X))
    print("Total features per sample:", X.shape[1])

    # Save extracted features
    feature_columns = [
        f"MFCC_{i + 1}" for i in range(X.shape[1])
    ]

    feature_df = pd.DataFrame(X, columns=feature_columns)
    feature_df["Emotion"] = y
    feature_df.to_csv("audio_features.csv", index=False)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    with open("results_summary.txt", "w") as file:
        file.write("Speech Emotion Recognition Results\n\n")
        file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        file.write(report)
        file.write(
            "\n\nEducational project. Predictions are not "
            "reliable psychological assessments."
        )

    # Save confusion matrix
    labels_sorted = sorted(np.unique(y))

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=labels_sorted,
        display_labels=labels_sorted,
        xticks_rotation=45,
        cmap="Blues"
    )

    plt.title("Speech Emotion Recognition - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Save model and scaler
    import joblib

    joblib.dump(model, "emotion_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # Save model accuracy
    with open("accuracy.txt", "w") as file:
        file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")

    print("\nProject completed successfully!")
    print("Generated files:")
    print("- audio_features.csv")
    print("- results_summary.txt")
    print("- confusion_matrix.png")
    print("- accuracy.txt")
    print("- emotion_model.pkl")
    print("- scaler.pkl")


if __name__ == "__main__":
    main()