import os, sys
import pandas as pd

def countFile(dir): 
    tmp = 0
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, item)):
            tmp += 1
        else:
            tmp += countFile(os.path.join(dir, item))
    return tmp

train_csv = pd.DataFrame(columns = ["gen","sp","filename"])
path = "/mnt/Germany_Birdcall/Germany_Birdcall_resampled"
Dropping_Threshold = 10
primary_labels = os.listdir( path )
i = 0
j = 0
BIRD_CODE = {}
for gen in primary_labels:
    path_sub = path + "/" + gen
    if countFile(path_sub) >= Dropping_Threshold:
        BIRD_CODE[gen] = j
        j = j + 1
        secondary_labels = os.listdir(path_sub)
        for sp in secondary_labels:
          path_sub_sub = path_sub + "/" + sp
          filenames = os.listdir(path_sub_sub)
          for filename in filenames:
              train_csv.loc[i] = [gen, sp, filename]
              i = i + 1
train_csv.to_csv('Germany_Birdcall_resampled_filtered.csv', index=False)
print(BIRD_CODE)
