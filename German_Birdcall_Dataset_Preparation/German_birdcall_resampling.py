import argparse
import soundfile as sf
import warnings

import librosa
import pandas as pd

from pathlib import Path
from joblib import delayed, Parallel


def resample(df: pd.DataFrame, target_sr: int):
    audio_dir = Path("./")
    resample_dir = Path("/mnt/Germany_Birdcall/Germany_Birdcall_resampled")
    resample_dir.mkdir(exist_ok=True, parents=True)
    warnings.simplefilter("ignore")

    for i, row in df.iterrows():
        gen = row["gen"]
        sp = row["sp"]
        filename = row["file-name"]
        primary_dir = resample_dir / gen
        if not primary_dir.exists():
            primary_dir.mkdir(exist_ok=True, parents=True)        
        secondary_dir = primary_dir / sp
        if not secondary_dir.exists():
            secondary_dir.mkdir(exist_ok=True, parents=True)

        try:
            y, _ = librosa.load(
                audio_dir / gen / sp / filename,
                sr=target_sr, mono=True, res_type="kaiser_fast")

            filename = filename.replace(".mp3", ".wav")
            sf.write(secondary_dir / filename, y, samplerate=target_sr)
        except Exception:
            with open("skipped.txt", "a") as f:
                file_path = str(audio_dir / gen / sp / filename)
                f.write(file_path + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", default=32000, type=int)
    parser.add_argument("--n_splits", default=20, type=int)
    args = parser.parse_args()

    target_sr = args.sr

    train = pd.read_csv("xeno_canto_sounds_germany.csv")
    dfs = []
    for i in range(args.n_splits):
        if i == args.n_splits - 1:
            start = i * (len(train) // args.n_splits)
            df = train.iloc[start:, :].reset_index(drop=True)
            dfs.append(df)
        else:
            start = i * (len(train) // args.n_splits)
            end = (i + 1) * (len(train) // args.n_splits)
            df = train.iloc[start:end, :].reset_index(drop=True)
            dfs.append(df)

    Parallel(
        n_jobs=args.n_splits,
        verbose=10)(delayed(resample)(df, args.sr) for df in dfs)
    
    print("All files' resampling finish.")
