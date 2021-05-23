# German Birdcall Classification in Soundscape with PANNs CNN series

This repo contains code for the research internship topic: German Birdcall Classification in Soundscape, it's developed from the origional repo of PANNs provided in https://github.com/qiuqiangkong/audioset_tagging_cnn. 

## Environments
The codebase is developed with Python 3.7. Install requirements as follows:
```
pip install -r requirements.txt
```

## Download pretrained models of PANNs
Users can inference the tags of an audio recording using pretrained models without training. First, downloaded one pretrained model from https://zenodo.org/record/3987831, for example, the model named "Cnn14_mAP=0.431.pth". Then, execute the following commands to inference this [audio](resources/R9_ZSCveAHg_7s.wav):
```
CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
MODEL_TYPE="Cnn14"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py audio_tagging --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path="resources/R9_ZSCveAHg_7s.wav" --cuda
```

## Train PANNs from scratch
Users can train PANNs from scratch by executing the commands in runme.sh. The runme.sh consists of three parts. 1. Download the full dataset. 2. Pack downloaded wavs to hdf5 file to speed up loading. 3. Train PANNs. 


## 1. Download German Birdcall dataset from Xeno-Canto Community


## 2. Train
Training is easy!

## Fine-tune PANNs CNN models on German Birdcall Classification
After downloading the pretrained models. Build fine-tuned systems for new tasks is simple!

```
MODEL_TYPE="Transfer_Cnn14"
CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/finetune_template.py train --sample_rate=32000 --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=14000 --model_type=$MODEL_TYPE --pretrained_checkpoint_path=$CHECKPOINT_PATH --cuda
```

## FAQs
If users came across out of memory error, then try to reduce the batch size.


## Reference
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).
