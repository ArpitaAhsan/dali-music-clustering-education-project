 import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

BASE_PATH = '/content/drive/MyDrive/425_DALI'
CSV_PATH = f'{BASE_PATH}/fma_style_dali_6000.csv'

def extract_mfcc(path):
    try:
        y, sr = librosa.load(path, sr=22050, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        return np.concatenate([mfcc.mean(axis=1), delta.mean(axis=1)])
    except:
        return np.zeros(26)

def load_dataset():
    df = pd.read_csv(CSV_PATH)
    df = df[df['filepath'].apply(os.path.exists)]
    print(f" {len(df)} tracks loaded")
    return df

def get_mfcc_features(df):
    print("Extracting MFCC features")
    X_mfcc = np.array([extract_mfcc(f) for f in df['filepath']])
    print(f"Raw features shape: {X_mfcc.shape}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mfcc)
    return X_scaled, scaler

