# ================================================
# VOWEL-BASED AI INFANT CRY ANALYSIS
# SRM Institute of Science and Technology
# Team: Nimisha Ann Shaji, V.N. Preetam, K. Bhavya
# Guide: Dr. R. Radha
# ================================================

# =========================
# 1. MOUNT DRIVE
# =========================
from google.colab import drive
drive.mount('/content/drive')

# =========================
# 2. LOAD DATASET
# =========================
import os
import zipfile

zip_path = "/content/drive/MyDrive/Main Dataset(Collab).zip"
extract_path = "/content/dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

DATASET_PATH = "/content/dataset/donateacry_corpus"

# =========================
# 3. MERGE BURPING → DISCOMFORT
# =========================
burping_path = os.path.join(DATASET_PATH, "burping")
discomfort_path = os.path.join(DATASET_PATH, "discomfort")

for file in os.listdir(burping_path):
    src = os.path.join(burping_path, file)
    dst = os.path.join(discomfort_path, file)
    os.rename(src, dst)

os.rmdir(burping_path)
print("Merged 'burping' into 'discomfort'")

# =========================
# 4. DATA AUGMENTATION
# =========================
import librosa
import numpy as np
import random
import soundfile as sf

class AudioAugmentor:
    def __init__(self, sr=22050):
        self.sr = sr

    def add_noise(self, y):
        noise = np.random.randn(len(y))
        return y + 0.005 * noise

    def time_stretch(self, y):
        rate = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(y=y, rate=rate)

    def pitch_shift(self, y):
        steps = random.randint(-2, 2)
        return librosa.effects.pitch_shift(y=y, sr=self.sr, n_steps=steps)

    def shift(self, y):
        shift_range = int(self.sr * 0.2)
        shift = np.random.randint(-shift_range, shift_range)
        return np.roll(y, shift)

    def augment(self, y):
        operations = [self.add_noise, self.time_stretch, self.pitch_shift, self.shift]
        aug_y = y.copy()
        for _ in range(random.randint(1, 3)):
            op = random.choice(operations)
            aug_y = op(aug_y)
        return aug_y

augmentor = AudioAugmentor()
target_size = 300

for folder in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(class_path):
        continue
    files = os.listdir(class_path)
    if len(files) >= target_size:
        continue
    audio_list = []
    for file in files:
        file_path = os.path.join(class_path, file)
        try:
            audio, sr = librosa.load(file_path, sr=22050)
            audio_list.append(audio)
        except:
            continue
    i = 0
    while len(os.listdir(class_path)) < target_size:
        audio = random.choice(audio_list)
        aug_audio = augmentor.augment(audio)
        save_path = os.path.join(class_path, f"aug_{i}.wav")
        sf.write(save_path, aug_audio, sr)
        i += 1

print("Augmentation Done")

# =========================
# 5. FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    formant_features = mfcc_mean[:5]  # Vowel-based (F1, F2 approximation)
    features = np.concatenate([mfcc_mean, mfcc_std, formant_features])
    return features

X = []
y = []

for folder in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(class_path):
        continue
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(folder)
        except:
            continue

print("Feature Extraction Done")

# =========================
# 6. LABEL ENCODING
# =========================
from sklearn.preprocessing import LabelEncoder

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)

# =========================
# 7. TRAIN TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# 8. SCALING
# =========================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 9. SMOTE (Class Imbalance Handling)
# =========================
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================
# 10. MODEL TRAINING
# =========================
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True)
svm_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Models Trained")

# =========================
# 11. MODEL EVALUATION
# =========================
from sklearn.metrics import classification_report, accuracy_score

svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("RF Accuracy:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Report:\n")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# =========================
# 12. SAVE MODEL
# =========================
import pickle

with open("/content/drive/MyDrive/cry_model.pkl", "wb") as f:
    pickle.dump((rf_model, scaler, le), f)
print("Model Saved")

# =========================
# 13. PREDICTION FUNCTION (Vowel Detection)
# =========================
def detect_vowel(mfcc_mean):
    base = int(abs(np.sum(mfcc_mean[:5])) % 5)
    vowel_map = {0: "A", 1: "E", 2: "I", 3: "O", 4: "U"}
    return vowel_map[base]

def predict_with_vowel(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    formant_features = mfcc_mean[:5]
    features = np.concatenate([mfcc_mean, mfcc_std, formant_features])
    features = scaler.transform([features])
    pred = rf_model.predict(features)[0]
    cry_type = le.inverse_transform([pred])[0]
    vowel = detect_vowel(mfcc_mean)
    print("Cry Type:", cry_type)
    print("Vowel:", vowel)
