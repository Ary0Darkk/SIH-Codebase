import os
import glob
import numpy as np
import pandas as pd
import joblib

from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

import warnings
warnings.filterwarnings("ignore")


# FEATURE EXTRACTION

def extract_features(window):
    x = np.array(window)
    N = len(x)

    # Time-domain
    mean = np.mean(x)
    std = np.std(x)
    var = np.var(x)
    rms = np.sqrt(np.mean(x**2))
    mav = np.mean(np.abs(x))
    wl = np.sum(np.abs(np.diff(x)))
    zc = np.sum(np.diff(np.sign(x)) != 0)

    # "Slope sign change"
    ssc = np.sum(np.diff(np.sign(np.diff(x))) != 0)

    # Frequency-domain
    yf = np.abs(rfft(x))
    xf = rfftfreq(N, d=1)

    spectral_energy = np.sum(yf**2)
    spectral_centroid = np.sum(xf * yf) / (np.sum(yf) + 1e-8)
    spectral_entropy = entropy((yf + 1e-8) / np.sum(yf))

    return [
        mean, std, var, rms, mav,
        wl, zc, ssc,
        spectral_energy, spectral_centroid, spectral_entropy
    ]


# LOAD DATA + SLIDING WINDOWS

def load_dataset(folder="emg_dataset", win_size=50, step=25):

    X_feat, X_raw, y = [], [], []

    # files = glob.glob(os.path.join(folder, "*.csv"))
    # print(f"\n📂 Found {len(files)} CSV gesture files\n")
    
    files = ['raw_data/round.csv','raw_data/shoot.csv','raw_data/up_down.csv']

    for file in files:
        df = pd.read_csv(file)
        gesture = os.path.basename(file).split(".")[0]
        print(f"📥 Loaded: {gesture}")

        sig = df["emg_value"].values

        for i in range(0, len(sig) - win_size, step):
            win = sig[i:i+win_size]

            X_feat.append(extract_features(win))
            X_raw.append(win)
            y.append(gesture)

    X_feat = np.array(X_feat)
    X_raw = np.array(X_raw)
    y = np.array(y)

    print(f"\n📊 Total window samples: {len(X_feat)}")

    return X_feat, X_raw, y


# TRAIN ML MODELS

def evaluate_ml_models(X, y, cv_folds):
    models = {
        "RandomForest": RandomForestClassifier(300),
        "SVM": SVC(kernel="rbf"),
        "KNN": KNeighborsClassifier(3),
        "LogisticRegression": LogisticRegression(max_iter=3000)
    }

    results = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        score = cross_val_score(pipe, X, y, cv=cv_folds).mean()
        results[name] = score

        print(f"{name} Accuracy → {score:.4f}")

    return results


# 1D-CNN MODEL

def build_cnn(input_len, num_classes):

    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(32, kernel_size=5, padding = 'same',activation="relu"),
        layers.MaxPooling1D(),
        layers.Conv1D(64, kernel_size=5, padding='same',activation="relu"),
        layers.MaxPooling1D(),
        layers.Dropout(0.1),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def evaluate_cnn(X_raw, y, epochs=35):
    X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)

    # Encode labels
    labels = sorted(list(np.unique(y)))
    label_to_int = {l: i for i, l in enumerate(labels)}
    y_int = np.array([label_to_int[x] for x in y])

    model = build_cnn(input_len=X_raw.shape[1], num_classes=len(labels))

    history = model.fit(
        X_raw, y_int,
        epochs=epochs,
        batch_size=8,
        verbose=0
    )

    acc = history.history["accuracy"][-1]
    print(f"✔ 1D-CNN Accuracy → {acc:.4f}")

    model.save("models/best_cnn_model.keras")
    print("💾 Saved CNN → best_cnn_model.keras")

    return acc


# LSTM MODEL (powerful for EMG time series)

def build_lstm(input_len, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.1),
        layers.LSTM(128,return_sequences=True),
        layers.Dropout(0.1),
        layers.LSTM(128),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.05),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def evaluate_lstm(X_raw, y, epochs=40):
    X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)

    labels = sorted(list(np.unique(y)))
    label_to_int = {l: i for i, l in enumerate(labels)}
    y_int = np.array([label_to_int[x] for x in y])

    model = build_lstm(input_len=X_raw.shape[1], num_classes=len(labels))

    history = model.fit(
        X_raw, y_int,
        epochs=epochs,
        batch_size=8,
        verbose=0
    )

    acc = history.history["accuracy"][-1]
    print(f"✔ LSTM Accuracy → {acc:.4f}")

    model.save("models/best_lstm_model.keras")
    print("💾 Saved LSTM → best_lstm_model.keras")

    return acc



if __name__ == "__main__":

    print("\n==================================================")
    print("            EMG SUPER TRAINER (ML + CNN + LSTM)")
    print("==================================================\n")

    X_feat, X_raw, y = load_dataset()

    min_class = min(pd.Series(y).value_counts())
    cv_folds = max(2, min(5, min_class))

    print(f"\n🔍 Using CV={cv_folds}\n")

    # ML Models
    print("🔥 Training ML Models...")
    ml_results = evaluate_ml_models(X_feat, y, cv_folds)

    # CNN
    print("\n🔥 Training CNN...")
    cnn_acc = evaluate_cnn(X_raw, y, epochs=40)
    ml_results["CNN"] = cnn_acc

    # LSTM
    print("\n🔥 Training LSTM...")
    lstm_acc = evaluate_lstm(X_raw, y, epochs=40)
    ml_results["LSTM"] = lstm_acc

    # Best Model
    best_model = max(ml_results, key=ml_results.get)
    print(f"\n🏆 BEST MODEL → {best_model} (Accuracy={ml_results[best_model]:.4f})")

    # Save ML model if best
    if best_model not in ["CNN", "LSTM"]:
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", {
                "RandomForest": RandomForestClassifier(300),
                "SVM": SVC(kernel="rbf", probability=True),
                "KNN": KNeighborsClassifier(3),
                "LogisticRegression": LogisticRegression(max_iter=3000)
            }[best_model])
        ])
        final_model.fit(X_feat, y)
        joblib.dump(final_model, "models/best_ml_model.pkl")
        print("💾 Saved best ML model → best_ml_model.pkl")

    print("\n🎉 TRAINING COMPLETE!")
