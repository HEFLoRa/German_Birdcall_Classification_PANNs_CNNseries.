import soundfile as sf
import librosa
import pandas as pd

train_csv = pd.read_csv("Germany_Birdcall_resampled_filtered.csv")
datadir = "/mnt/GermanyBirdcall/Germany_Birdcall_resampled"

idx = 19783
sample = train_csv.loc[idx, :]
wav_name = sample["filename"]
gen = sample["gen"]
sp = sample["sp"]
# file_path = self.datadir / ebird_code / mp3_name
file_path = datadir + "/" + gen + "/"+ sp + "/" + wav_name

# y = librosa.core.load(file_path, SR)[0]
y, sr = sf.read(file_path)
print(y)
