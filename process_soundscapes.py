import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import sys, time, glob
import numpy as np
import polars as pl
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm
from DataProcessing import extract_optimized_features

SOUNDSCAPES_DIR = "Data/train_soundscapes/"
OUTPUT_PATH     = "features_soundscapes_158.parquet"

# Centre exact des coordonnées du Pantanal
DEFAULT_LAT     = -19.05
DEFAULT_LON     = -56.75


def save_parquet(all_rows, output_path):
    """Éclate les 158 features en colonnes et sauvegarde en Parquet."""
    if not all_rows:
        print("Aucune donnée à sauvegarder.")
        return
    df = pl.DataFrame(all_rows)
    df = df.with_columns(
        [pl.col("features").list.get(i).alias(f"feat_{i}") for i in range(158)]
    ).drop("features")
    df.write_parquet(output_path)
    print(f"\n✅ {len(df)} segments sauvegardés → {output_path} (shape: {df.shape})")


def warmup_numba():
    """Vérifie le bon fonctionnement de l'extraction."""
    print("Vérification de l'environnement...", end=" ", flush=True)
    dummy = np.random.randn(32000).astype(np.float32)
    extract_optimized_features(dummy, sr=32000)
    print("OK (Mode sans JIT activé)")


def process_audio_file(file_path, sr=32000, segment_dur=5):
    """Charge l'audio et le découpe dynamiquement en segments de 5s."""
    try:
        y, _ = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"\nErreur de chargement {file_path}: {e}")
        return []

    filename = os.path.basename(file_path)
    segment_len = sr * segment_dur
    rows = []

    # Découpage par fenêtres de 5 secondes non-chevauchantes
    for start_sample in range(0, len(y) - segment_len + 1, segment_len):
        start_sec = start_sample / sr
        end_sec = start_sec + segment_dur
        segment = y[start_sample : start_sample + segment_len]

        try:
            features = extract_optimized_features(segment, sr=sr)
            rows.append({
                "filename": filename,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "latitude": DEFAULT_LAT,
                "longitude": DEFAULT_LON,
                "features": features.tolist()
            })
        except Exception as e:
            print(f"\nErreur extraction {filename} ({start_sec}s-{end_sec}s): {e}")

    # Libération explicite et immédiate de la RAM pour les gros tableaux NumPy
    del y
    return rows



def main():
    if not os.path.exists(SOUNDSCAPES_DIR):
        sys.exit(f" Dossier introuvable : {SOUNDSCAPES_DIR}")

    # Recherche de tous les fichiers audios (.ogg, .mp3, .wav)
    audio_files = []
    for ext in ["ogg", "mp3", "wav", "OGG", "MP3", "WAV"]:
        audio_files.extend(glob.glob(os.path.join(SOUNDSCAPES_DIR, f"*.{ext}")))
    
    audio_files = sorted(audio_files)
    total_files = len(audio_files)
    
    if total_files == 0:
        sys.exit(f" Aucun fichier audio trouvé dans {SOUNDSCAPES_DIR}")

    print(f"Trouvé : {total_files} fichiers audio.")
    warmup_numba()

    # Préparation des tâches pour le parallélisme
    tasks = (delayed(process_audio_file)(fp) for fp in audio_files)

    all_rows = []
    t0 = time.time()
    try:
        # Exécution en parallèle avec joblib et affichage progressif avec tqdm
        gen = Parallel(n_jobs=-1, backend="loky", return_as="generator")(tasks)
        for results in tqdm(gen, total=total_files, desc="Extraction", unit=" fichier"):
            all_rows.extend(results)
    except KeyboardInterrupt:
        print(f"\n  Interruption détectée ! Sauvegarde des {len(all_rows)} segments déjà extraits...")
    finally:
        save_parquet(all_rows, OUTPUT_PATH)
        print(f"Temps total : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
