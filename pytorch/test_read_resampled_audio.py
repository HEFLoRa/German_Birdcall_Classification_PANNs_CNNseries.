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
file_path = '/mnt/Germany_Birdcall/Germany_Birdcall_resampled/Merops/apiaster/XC500595-190914 Bienenfresser Zugruf FH 1701.wav'
# y = librosa.core.load(file_path, SR)[0]
y, sr = sf.read(file_path)
print(y)
