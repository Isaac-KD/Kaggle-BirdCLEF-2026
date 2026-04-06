import polars as pl
import librosa
import numpy as np
from skimage.filters import median
from skimage.morphology import disk
import librosa
import numpy as np
import polars as pl
from tqdm import tqdm
from itertools import groupby

def load_and_clean_dataset(csv_path, audio_dir):
    """
    Charge, nettoie et convertit le dataset audio en format Parquet.
    """

    print(f"--- Traitement du CSV : {csv_path} ---")
    # 1. Lecture avec schéma strict pour éviter les erreurs de type (ashgre1, etc.)
    df = pl.read_csv(
        csv_path,
        schema_overrides={
            "primary_label": pl.String,
            "secondary_labels": pl.String,
            "type": pl.String,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "rating": pl.Float64,
            "filename": pl.String
        },
        infer_schema_length=20000
    )

    # 2. Pipeline de transformation
    df = (
        df.drop(["url", "license", "author"])
        .with_columns(
            full_path = pl.lit(audio_dir) + pl.col("filename")
        )
        .drop("filename")
    )

    return df

def get_audio_vector(path, target_length=None):
    """Charge l'audio et retourne un array numpy.
    target_length : nombre d'échantillons pour compléter/tronquer si nécessaire
    """
    y, sr = librosa.load(path, sr=32000)
    if target_length is not None:
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
    return y, sr

def clean_spectrogram(S_db):
    img = (S_db - S_db.min()) / (S_db.max() - S_db.min()+1e-10)
    img = median(img, disk(1))
    threshold = np.median(img) * 1.5
    img[img < threshold] = 0
    return img

def get_mel(path):
    y, sr = librosa.load(path, sr=32000)  # charge tout l'audio
    audio_dur = len(y) / sr  # durée en secondes
    #y = bandpass_filter(y, sr, 1000, 10000)

    S = librosa.feature.melspectrogram(y=y, sr=sr,n_mels=128,n_fft=1024)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = clean_spectrogram(S_dB)
    return S_dB,sr,audio_dur


######### Optional ##############

def ic(window):
    # Fréquence dominante par frame
    dominant_freqs = np.argmax(window, axis=0)  # shape (n_frames,)
    n = len(dominant_freqs)
    if n < 2: return 0
    
    # Compte les occurrences de chaque fréquence
    counts = np.bincount(dominant_freqs, minlength=window.shape[0])
    
    # IC = sum(ni * (ni-1)) / (n * (n-1))
    return np.sum(counts * (counts - 1)) / (n * (n - 1))

def compute_ic_df(df, columns="class_name", window_sec=1, sr=32000, hop_length=512):
    frames_per_sec = int(sr / hop_length)
    step = frames_per_sec * window_sec
    rows = []
    
    for name in np.unique(df[columns].to_list()):
        paths = df.filter(df[columns] == name)["full_path"].to_list()
        
        for path in paths:
            try:
                S_dB, sr, audio_dur = get_mel(path)
                S_clean = clean_spectrogram(S_dB)
                
                for t in range(0, S_clean.shape[1] - step, step):
                    window = S_clean[:, t:t + step]
                    rows.append({
                        "class_name": name,
                        "file": path,
                        "t_sec": t / frames_per_sec,
                        "IC": ic(window)
                    })
            except: continue
    
    return pl.DataFrame(rows)

######################################

def extract_mfcc(y, sr, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc.max(axis=1),
        mfcc.min(axis=1)
    ])

#####################################


def process_file(path, label, sr=32000, segment_dur=5):
    y, sr = librosa.load(path, sr=sr)
    segment_len = sr * segment_dur
    rows = []
    for start in range(0, len(y) - segment_len, segment_len):
        segment = y[start:start + segment_len]
        features = extract_mfcc(segment, sr)
        rows.append({"common_name": label, "features": features.tolist()})
    return rows

# ==========================================
# 1. FONCTIONS DE NETTOYAGE ET UTILITAIRES
# ==========================================

def clean_spectrogram(S_db):
    """ Filtre médian et seuillage pour débruiter le spectrogramme """
    img = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-10)
    img = median(img, disk(1))
    threshold = np.median(img) * 1.5
    img[img < threshold] = 0
    return img

def safe_stat(array, stat_func, default=0.0):
    """ Évite les erreurs sur les arrays vides (ex: pas d'oiseau détecté) """
    if len(array) == 0 or np.all(np.isnan(array)):
        return default
    return stat_func(array)

# ==========================================
# 2. L'EXTRACTEUR OPTIMISÉ (158 FEATURES)
# ==========================================

def extract_optimized_features(y, sr=32000):
    """
    Extrait les 158 features ultra-discriminantes pour une fenêtre audio (ex: 5s).
    Renvoie un array numpy 1D de taille (158,).
    """
    features =[]
    
    # --- PRÉCALCULS GLOBAUX (Pour gagner du temps) ---
    # On calcule la STFT et le Mel une seule fois !
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    mel = librosa.feature.melspectrogram(S=stft**2, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    clean_mel = clean_spectrogram(mel_db)
    
    # --- 1. MFCCs (80) & Deltas (26) ---
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    features.extend([
        mfcc.mean(axis=1), mfcc.std(axis=1), 
        mfcc.max(axis=1), mfcc.min(axis=1)
    ]) # 4 x 20 = 80
    
    delta_mfcc = librosa.feature.delta(mfcc[:13])
    features.extend([
        delta_mfcc.std(axis=1), np.max(np.abs(delta_mfcc), axis=1)
    ]) # 2 x 13 = 26
    
    # --- 2. Features Spectrales Globales (2 + 14 + 2 + 2 = 20) ---
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    features.extend([safe_stat(centroid, np.mean), safe_stat(centroid, np.std)]) # 2
    
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=6) # 7 bandes (6 + 1)
    features.extend([contrast.mean(axis=1), contrast.std(axis=1)]) # 14
    
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    features.extend([safe_stat(flatness, np.mean), safe_stat(flatness, np.std)]) # 2
    
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.extend([safe_stat(zcr, np.mean), safe_stat(zcr, np.std)]) # 2
    
    # --- 3. Pitch F0 (4) & Modulation FM (3) ---
    # YIN est le meilleur algo de pitch. On le limite[200Hz - 10000Hz] (zone oiseaux) pour aller plus vite
    f0 = librosa.yin(y, fmin=200, fmax=10000, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    
    if len(f0_clean) > 0:
        features.extend([np.mean(f0_clean), np.std(f0_clean), np.min(f0_clean), np.max(f0_clean)]) # 4
        fm = np.diff(f0_clean)
        features.extend([safe_stat(fm, np.mean), safe_stat(fm, np.std), safe_stat(np.abs(fm), np.max)]) # 3
    else:
        features.extend([0]*7) # Remplit de zéros si pas de pitch
        
    # --- 4. Stats Mel Spectrogramme (16) ---
    # Diviser le mel en 8 sous-bandes fréquentielles
    mel_bands = np.array_split(clean_mel, 8, axis=0)
    for band in mel_bands:
        features.extend([np.mean(band), np.std(band)]) # 16 (8x2)

    # --- 5. Rythme, Syllabes et Active Frames (1 + 5 = 6) ---
    # Une frame est active si elle n'a pas été mise à 0 par clean_spectrogram
    active_frames = (clean_mel.sum(axis=0) > 0).astype(int)
    features.append(np.mean(active_frames)) # 1 (Proportion active)
    
    frame_dur = 512 / sr # Durée d'une frame
    syllables =[sum(1 for _ in g) for v, g in groupby(active_frames) if v == 1]
    silences =[sum(1 for _ in g) for v, g in groupby(active_frames) if v == 0]
    
    features.extend([
        len(syllables),                                              # nb syllabes
        safe_stat(syllables, np.mean) * frame_dur,                   # durée mean
        safe_stat(syllables, np.std) * frame_dur,                    # durée std
        safe_stat(silences, np.mean) * frame_dur,                    # espace mean
        safe_stat(silences, np.std) * frame_dur                      # espace std
    ]) # 5

    # --- 6. Plage Fréquentielle Active (3) ---
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)
    active_bins = np.where(clean_mel.sum(axis=1) > 0)[0]
    
    if len(active_bins) > 0:
        f_min, f_max = mel_freqs[active_bins[0]], mel_freqs[active_bins[-1]]
        features.extend([f_min, f_max, f_max - f_min]) # 3
    else:
        features.extend([0, 0, 0])
        
    # Flatten the result into a strict 1D numpy array of 158 elements
    final_vector = np.concatenate([np.array([f]).flatten() for f in features])
    return final_vector.astype(np.float32)

def load_data(audio_path="features_birdclef_158.parquet", 
                          soundscapes_path="features_soundscapes_158.parquet"):
    """
    Charge, aligne et fusionne les datasets Parquet pour générer les matrices X et y.
    
    Retourne :
        X (np.ndarray): Matrice des features (latitude, longitude, 158 features audio).
        y (np.ndarray): Matrice des labels (One-Hot Encoding).
        df_train (pl.DataFrame): Le DataFrame Polars combiné complet.
    """
    print("1. Chargement des datasets...")
    df_audio = pl.read_parquet(audio_path)
    df_soundscapes = pl.read_parquet(soundscapes_path)

    print(f"Shape initial - Audio: {df_audio.shape}, Soundscapes: {df_soundscapes.shape}")

    # 2. Définition des colonnes à garder
    # On prend les 158 features + la position géo + les labels + le nom du fichier pour le suivi
    feature_cols = [f"feat_{i}" for i in range(158)]
    common_cols = ["filename", "latitude", "longitude", "labels"] + feature_cols

    # 3. Alignement et Fusion
    print("\n2. Alignement et fusion des données...")
    # On sélectionne uniquement les colonnes communes pour éviter les erreurs de concaténation
    df_audio_clean = df_audio.select(common_cols)
    df_soundscapes_clean = df_soundscapes.select(common_cols)

    # Concaténation verticale
    df_train = pl.concat([df_audio_clean, df_soundscapes_clean])
    print(f"✅ Fusion réussie ! Shape final: {df_train.shape}")

    # 4. Préparation de X (Features) et y (Cibles)
    print("\n3. Création des matrices X et y pour l'entraînement...")

    # X : Nos 158 features audio + la latitude et la longitude
    cols_for_X = ["latitude", "longitude"] + feature_cols
    X = df_train.select(cols_for_X).to_numpy()

    # y : On empile les listes (One-Hot Encoding) 
    y = np.vstack(df_train["labels"].to_list())

    print(f"Matrice X (Features) prête : {X.shape}")
    print(f"Matrice y (Labels) prête   : {y.shape}")
    
    return X, y, df_train