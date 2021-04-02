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
    'Garrulus': 0, 'Circus': 1, 'Sylvia': 2, 'Coccothraustes': 3, 'Chroicocephalus': 4,
    'Pluvialis': 5, 'Ixobrychus': 6, 'Columba': 7, 'Psittacula': 8, 'Glaucidium': 9, 'Jynx': 10,
    'Crex': 11, 'Remiz': 12, 'Aix': 13, 'Linaria': 14, 'Locustella': 15, 'Calidris': 16,
    'Aythya': 17, 'Cuculus': 18, 'Lullula': 19, 'Recurvirostra': 20, 'Spatula': 21, 'Chlidonias': 22,
    'Ardea': 23, 'Phylloscopus': 24, 'Larus': 25, 'Streptopelia': 26, 'Limosa': 27, 'Lanius': 28,
    'Oecanthus': 29, 'Scolopax': 30, 'Phalacrocorax': 31, 'Perdix': 32, 'Caprimulgus': 33,
    'Grus': 34, 'Botaurus': 35, 'Gallinago': 36, 'Cyanistes': 37, 'Dendrocoptes': 38, 'Hirundo': 39,
    'Chloris': 40, 'Tadorna': 41, 'Delichon': 42, 'Strix': 43, 'Oriolus': 44, 'Pica': 45,
    'Haliaeetus': 46, 'Sonus': 47, 'Numenius': 48, 'Porzana': 49, 'Passer': 50, 'Rallus': 51,
    'Spinus': 52, 'Poecile': 53, 'Dryocopus': 54, 'Sturnus': 55, 'Bubo': 56, 'Riparia': 57,
    'Upupa': 58, 'Gallinula': 59, 'Otus': 60, 'Dendrocopos': 61, 'Loxia': 62, 'Podiceps': 63,
    'Dryobates': 64, 'Alauda': 65, 'Prunella': 66, 'Anthus': 67, 'Panurus': 68, 'Fulica': 69,
    'Carduelis': 70, 'Charadrius': 71, 'Amazona': 72, 'Apus': 73, 'Haematopus': 74, 'Carpodacus': 75,
    'Mystery': 76, 'Alopochen': 77, 'Falco': 78, 'Fringilla': 79, 'Anser': 80, 'Mareca': 81,
    'Actitis': 82, 'Motacilla': 83, 'Alcedo': 84, 'Mergus': 85, 'Tachybaptus': 86, 'Corvus': 87,
    'Vanellus': 88, 'Ficedula': 89, 'Cygnus': 90, 'Saxicola': 91, 'Bucephala': 92, 'Bombycilla': 93,
    'Hippolais': 94, 'Branta': 95, 'Sitta': 96, 'Emberiza': 97, 'Regulus': 98, 'Tyto': 99,
    'Muscicapa': 100, 'Luscinia': 101, 'Lophophanes': 102, 'Erithacus': 103, 'Phasianus': 104,
    'Milvus': 105, 'Anas': 106, 'Acanthis': 107, 'Picus': 108, 'Aegolius': 109, 'Certhia': 110,
    'Periparus': 111, 'Aegithalos': 112, 'Sterna': 113, 'Buteo': 114, 'Serinus': 115, 'Phoenicurus': 116,
    'Cinclus': 117, 'Turdus': 118, 'Troglodytes': 119, 'Athene': 120, 'Tringa': 121, 'Coturnix': 122,
    'Merops': 123, 'Coloeus': 124, 'Accipiter': 125, 'Pyrrhula': 126, 'Nycticorax': 127, 'Asio': 128,
    'Acrocephalus': 129, 'Parus': 130
}

# BIRD_CODE = {  # if not filtered
#     'Pandion': 0, 'Iduna': 1, 'Lyrurus': 2, 'Aquila': 3, 'Garrulus': 4, 'Circus': 5,
#     'Sylvia': 6, 'Coccothraustes': 7, 'Clangula': 8, 'Pernis': 9, 'Nymphicus': 10,
#     'Burhinus': 11, 'Chroicocephalus': 12, 'Pluvialis': 13, 'Ixobrychus': 14, 'Columba': 15,
#     'Metrioptera': 16, 'Somateria': 17, 'Plectrophenax': 18, 'Stethophyma': 19, 'Psittacula': 20,
#     'Tetrao': 21, 'Glaucidium': 22, 'Jynx': 23, 'Crex': 24, 'Conocephalus': 25,
#     'Gryllus': 26, 'Himantopus': 27, 'Remiz': 28, 'Aix': 29, 'Linaria': 30,
#     'Locustella': 31, 'Calidris': 32, 'Aythya': 33, 'Cuculus': 34, 'Lullula': 35,
#     'Recurvirostra': 36, 'Spatula': 37, 'Thalasseus': 38, 'Chlidonias': 39, 'Tetrastes': 40,
#     'Ardea': 41, 'Phylloscopus': 42, 'Larus': 43, 'Streptopelia': 44, 'Limosa': 45,
#     'Lanius': 46, 'Oecanthus': 47, 'Scolopax': 48, 'Platalea': 49, 'Phalacrocorax': 50,
#     'Tarsiger': 51, 'Perdix': 52, 'Eremophila': 53, 'Caprimulgus': 54, 'Ciconia': 55,
#     'Hydroprogne': 56, 'Grus': 57, 'Otis': 58, 'Botaurus': 59, 'Gallinago': 60,
#     'Picoides': 61, 'Cyanistes': 62, 'Dendrocoptes': 63, 'Hirundo': 64, 'Chloris': 65,
#     'Galerida': 66, 'Tadorna': 67, 'Tachymarptis': 68, 'Delichon': 69, 'Strix': 70,
#     'Oriolus': 71, 'Pica': 72, 'Haliaeetus': 73, 'Sonus': 74, 'Chorthippus': 75,
#     'Numenius': 76, 'Lagopus': 77, 'Ichthyaetus': 78, 'Porzana': 79, 'Passer': 80,
#     'Rallus': 81, 'Ardeola': 82, 'Spinus': 83, 'Poecile': 84, 'Dryocopus': 85,
#     'Sturnus': 86, 'Bubo': 87, 'Riparia': 88, 'Gavia': 89, 'Upupa': 90,
#     'Calcarius': 91, 'Gallinula': 92, 'Otus': 93, 'Dendrocopos': 94, 'Loxia': 95,
#     'Podiceps': 96, 'Uria': 97, 'Dryobates': 98, 'Alauda': 99, 'Prunella': 100,
#     'Anthus': 101, 'Panurus': 102, 'Fulica': 103, 'Rhea': 104, 'Netta': 105,
#     'Carduelis': 106, 'Cisticola': 107, 'Charadrius': 108, 'Amazona': 109, 'Apus': 110,
#     'Haematopus': 111, 'Carpodacus': 112, 'Rissa': 113, 'Mystery': 114, 'Nemobius': 115,
#     'Alopochen': 116, 'Morus': 117, 'Falco': 118, 'Fringilla': 119, 'Anser': 120,
#     'Mareca': 121, 'Actitis': 122, 'Motacilla': 123, 'Alcedo': 124, 'Mergus': 125,
#     'Arenaria': 126, 'Tachybaptus': 127, 'Corvus': 128, 'Vanellus': 129, 'Calandrella': 130,
#     'Pyrrhocorax': 131, 'Ficedula': 132, 'Cygnus': 133, 'Saxicola': 134, 'Bucephala': 135,
#     'Bombycilla': 136, 'Hippolais': 137, 'Branta': 138, 'Sitta': 139, 'Emberiza': 140,
#     'Regulus': 141, 'Tyto': 142, 'Pseudochorthippus': 143, 'Gryllotalpa': 144, 'Muscicapa': 145,
#     'Phoenicopterus': 146, 'Phaneroptera': 147, 'Luscinia': 148, 'Lophophanes': 149, 'Erithacus': 150,
#     'Phasianus': 151, 'Milvus': 152, 'Anas': 153, 'Acanthis': 154, 'Picus': 155,
#     'Aegolius': 156, 'Certhia': 157, 'Gelochelidon': 158, 'Periparus': 159, 'Melanitta': 160,
#     'Aegithalos': 161, 'Sterna': 162, 'Buteo': 163, 'Serinus': 164, 'Phoenicurus': 165,
#     'Cinclus': 166, 'Stercorarius': 167, 'Turdus': 168, 'Troglodytes': 169, 'Athene': 170,
#     'Montifringilla': 171, 'Tringa': 172, 'Coturnix': 173, 'Merops': 174, 'Oenanthe': 175,
#     'Coloeus': 176, 'Accipiter': 177, 'Pyrrhula': 178, 'Cettia': 179, 'Nucifraga': 180,
#     'Nycticorax': 181, 'Asio': 182, 'Acrocephalus': 183, 'Clanga': 184, 'Parus': 185
# }

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


class SpectrogramDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: str,
                 img_size=224):
        self.df = df
        self.datadir = datadir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        mp3_name = sample["filename"]
        ebird_code = sample["ebird_code"]

        # y, sr = sf.read(self.datadir / ebird_code / mp3_name)
        # file_path = self.datadir / ebird_code / mp3_name
        file_path = self.datadir + "/" + ebird_code + "/"+ mp3_name
        y = librosa.core.load(file_path, SR)[0]

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

        # transfer to Mel-Spectrogram
        melspec = librosa.feature.melspectrogram(y, sr=SR, **melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        # one-hot coding ? or integer indexing ?
        # labels = np.zeros(len(BIRD_CODE), dtype=int)
        # labels[BIRD_CODE[ebird_code]] = 1
        # integer indexing
        label = BIRD_CODE[ebird_code]

        return {
            "image": image,
            "target": label
        }


def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V
