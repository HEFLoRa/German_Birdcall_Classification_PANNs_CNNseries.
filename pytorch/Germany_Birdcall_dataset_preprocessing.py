# copy form
# import cv2
# import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data as data
import time
import logging

# generate from train.csv after filtering minor classes
BIRD_CODE = {  # filtered by Dropping_Threshold = 10
    'Oecanthus': 0, 'Bubo': 1, 'Garrulus': 2, 'Bucephala': 3, 'Gallinago': 4, 'Oriolus': 5, 
    'Columba': 6, 'Perdix': 7, 'Anas': 8, 'Amazona': 9, 'Passer': 10, 'Aegolius': 11, 
    'Streptopelia': 12, 'Sturnus': 13, 'Motacilla': 14, 'Panurus': 15, 'Phalacrocorax': 16, 
    'Coloeus': 17, 'Chlidonias': 18, 'Apus': 19, 'Actitis': 20, 'Rallus': 21, 'Branta': 22, 
    'Recurvirostra': 23, 'Nycticorax': 24, 'Falco': 25, 'Jynx': 26, 'Carduelis': 27, 'Haliaeetus': 28, 
    'Dendrocopos': 29, 'Buteo': 30, 'Ficedula': 31, 'Emberiza': 32, 'Corvus': 33, 'Regulus': 34, 
    'Pluvialis': 35, 'Sylvia': 36, 'Aegithalos': 37, 'Dryocopus': 38, 'Carpodacus': 39, 
    'Botaurus': 40, 'Phoenicurus': 41, 'Muscicapa': 42, 'Troglodytes': 43, 'Coturnix': 44, 
    'Otus': 45, 'Saxicola': 46, 'Mergus': 47, 'Ixobrychus': 48, 'Tadorna': 49, 'Psittacula': 50, 
    'Remiz': 51, 'Charadrius': 52, 'Larus': 53, 'Poecile': 54, 'Haematopus': 55, 'Cyanistes': 56, 
    'Crex': 57, 'Glaucidium': 58, 'Acrocephalus': 59, 'Coccothraustes': 60, 'Fulica': 61, 'Linaria': 62, 
    'Luscinia': 63, 'Hippolais': 64, 'Sitta': 65, 'Porzana': 66, 'Anthus': 67, 'Upupa': 68, 'Spatula': 69, 
    'Phylloscopus': 70, 'Strix': 71, 'Mareca': 72, 'Picus': 73, 'Riparia': 74, 'Dryobates': 75, 
    'Sonus': 76, 'Tyto': 77, 'Chloris': 78, 'Lullula': 79, 'Podiceps': 80, 'Ardea': 81, 'Circus': 82, 
    'Hirundo': 83, 'Pica': 84, 'Alopochen': 85, 'Chroicocephalus': 86, 'Numenius': 87, 'Accipiter': 88, 
    'Alcedo': 89, 'Grus': 90, 'Athene': 91, 'Aythya': 92, 'Caprimulgus': 93, 'Locustella': 94, 
    'Gallinula': 95, 'Lophophanes': 96, 'Delichon': 97, 'Phasianus': 98, 'Acanthis': 99, 'Parus': 100, 
    'Anser': 101, 'Scolopax': 102, 'Turdus': 103, 'Limosa': 104, 'Spinus': 105, 'Serinus': 106, 
    'Prunella': 107, 'Periparus': 108, 'Certhia': 109, 'Loxia': 110, 'Bombycilla': 111, 'Sterna': 112, 
    'Asio': 113, 'Merops': 114, 'Pyrrhula': 115, 'Lanius': 116, 'Fringilla': 117, 'Erithacus': 118, 
    'Aix': 119, 'Cuculus': 120, 'Calidris': 121, 'Tringa': 122, 'Tachybaptus': 123, 'Cinclus': 124, 
    'Dendrocoptes': 125, 'Milvus': 126, 'Vanellus': 127, 'Cygnus': 128, 'Alauda': 129
}

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

PERIOD = 5  # 5-second random clips
SR = 32000  # resampling frequency


class WaveformDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: str):
        self.df = df
        self.datadir = datadir
        # self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        gen = sample["gen"]
        sp = sample["sp"]

        # file_path = self.datadir / ebird_code / mp3_name
        file_path = self.datadir + "/" + gen + "/"+ sp + "/" + wav_name
        # y = librosa.core.load(file_path, SR)[0]
        y, sr = sf.read(file_path)  # value range (-1, 1)
        # random chopping or expending
        len_y = len(y)
        effective_length = SR * PERIOD
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        # waveform = int16_to_float32(float32_to_int16(y))
        waveform = 1/2 * (y + 1)  #value range tranferred to [0, 1]

        '''Move Mel-Spectrogram Transformation from data.Dataset to nn.Module'''
        # # transfer to Mel-Spectrogram
        # melspec = librosa.feature.melspectrogram(y, sr=SR, **melspectrogram_parameters)
        # melspec = librosa.power_to_db(melspec).astype(np.float32)
        #
        # image = mono_to_color(melspec)
        # height, width, _ = image.shape
        # image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        # image = np.moveaxis(image, 2, 0)
        # image = (image / 255.0).astype(np.float32)

        '''one-hot coding(BCEloss) ? or integer indexing(CrossEntropyLoss) ?'''
        # one-hot coding
        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[gen]] = 1
        target = labels.astype(np.float32)
        # integer indexing
        # label = BIRD_CODE[ebird_code]

        data_dict = {
            'audio_name': wav_name, 'waveform': waveform, 'target': target  # len(target) = classes_num
        }

        return data_dict


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)


class BaseSampler(object):
    def __init__(self, df, batch_size, random_seed):
        """Base class of train sampler.

        Args:
          df: pd.Dataframe
          batch_size: int
          random_seed: int
        """
        self.df = df
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Load target
        load_time = time.time()

        self.audios_num = self.df.shape[0]
        self.classes_num = len(BIRD_CODE)  # also len(df["gen"].unique())
        logging.info('Number of samples: {}'.format(self.audios_num))
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))


class RandomSampler(BaseSampler):
    def __init__(self, df, batch_size, random_seed=1234):
        """Random sampler. Batch_data loops over the whole training set.

        Args:
          df: pd.Dataframe
          batch_size: int
          random_seed: int
        """
        super(RandomSampler, self).__init__(df, batch_size, random_seed)

        self.indexes = np.arange(self.audios_num)

        # Shuffle indexes
        self.random_state.shuffle(self.indexes)

        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training.

        Returns:
          batch_indices
        """
        batch_size = self.batch_size

        while True:
            batch_indices = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)

                # If audio in black list then continue
                batch_indices.append(index)
                i += 1

            yield batch_indices
            
            
class BalancedSampler(BaseSampler):
    def __init__(self, df, batch_size, random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are equally
        sampled from different sound classes.

        Args:
          df: pd.Dataframe
          batch_size: int
          random_seed: int
        """
        super(BalancedSampler, self).__init__(df, batch_size, random_seed)

        # 1. count num of samples per class
        # 2. Training indexes of all sound classes. E.g.:
        #    [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.samples_num_per_class = np.zeros(self.classes_num)
        count_df = df["gen"].value_counts()
        self.indexes_per_class = []
        for class_idx in range(self.classes_num):
            self.samples_num_per_class[class_idx] = count_df[INV_BIRD_CODE[class_idx]]
            self.indexes_per_class.append(np.where(
                df["gen"] == INV_BIRD_CODE[class_idx])[0])

        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))

        # Shuffle indexes for each class
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])

        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch indicies for training.

        Returns:
          batch_indices: e.g.: [0,8,13,...]
        """
        batch_size = self.batch_size

        while True:
            batch_indices = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                index = self.indexes_per_class[class_id][pointer]

                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                batch_indices.append(index)
                i += 1
            yield batch_indices

    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class,
            'queue': self.queue,
            'pointers_of_classes': self.pointers_of_classes}
        return state

    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']


melspectrogram_parameters = {
    "n_mels": 64,
    "fmin": 50,
    "fmax": 14000
}


def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...}, 
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict
