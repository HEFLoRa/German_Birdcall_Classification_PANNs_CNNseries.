# German Birdcall Classification in Soundscape with PANNs CNN series

This repo contains code for the research internship topic: German Birdcall Classification in Soundscape, it's developed from the origional repo of PANNs provided in https://github.com/qiuqiangkong/audioset_tagging_cnn. 

## Environments
The codebase is developed with Python 3.7. Install requirements as follows:
```
pip install -r requirements.txt
```
## Download German Birdcall Audios
To prepare the German Birdcall Dataset, zoom into the German_Birdcall_Dataset_Preparation folder and run the following:
```
python import requests.py
python German_birdcall_resampling.py
python Germany_Birdcall_resampled_csv_generation.py
```

## Download pretrained models of PANNs
Pretrianed models can be downloaded from https://zenodo.org/record/3987831, for example, we want to download the model named "Cnn10_mAP=0.380.pth".
```
wget -O $"Cnn10_mAP=0.380.pth" https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth?download=1
```

## Fine-tune PANNs CNN models on German Birdcall Classification
After preparing the German Birdcall Dataset and downloading the pretrained model, run the following to start training (fine-tuning):
```
python finetune_template_for_Germany_Birdcall.py train --workspace .. --model_type "Transfer_Cnn10" --pretrained_checkpoint_path "Cnn10_mAP=0.380.pth" --balanced none --augmentation mixup --freeze_base_num 0 --early_stop 50000
```

## Result Analysis
To analyze the training process and the model performance, refer to Jupyter notebooks in the Results Analysis folder. Also, some trained models have also been placed there, they are called in the Germany_Birdcall_Engine_nonemixup.ipynb.

## FAQs
If users came across out of memory error, then try to reduce the batch size.

## Reference
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).
