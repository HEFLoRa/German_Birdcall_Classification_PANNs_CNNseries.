import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import pandas as pd
import argparse
import time
import logging

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from models import *
# import config
from utilities import (create_folder, get_filename, create_logging, Mixup,
                       StatisticsContainer)
from pytorch_utils import (move_data_to_device, count_parameters, count_flops,
                           do_mixup)
from data_generator import (AudioSetDataset, TrainSampler, BalancedTrainSampler,
                            AlternateTrainSampler, EvaluateSampler, collate_fn)
from evaluate import Evaluator
from losses import get_loss_func

# Germany Birdcall
from Germany_Birdcall_dataset_preprocessing import WaveformDataset, BalancedSampler
from sklearn.model_selection import StratifiedShuffleSplit


class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

#         clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        clipwise_output = torch.sigmoid(self.fc_transfer(embedding))
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict


def train(args):
    """ Arguments & parameters"""
    # from main.py
    workspace = args.workspace  # store experiments results in the workspace
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    # resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    filename = args.filename
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # for fine-tune models
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base

    pretrain = True if pretrained_checkpoint_path else False

    # Saving Paths
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
                                   'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                                       sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
                                    model_type,
                                   'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
                                   'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename,
                                   'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                                       sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
                                    model_type,
                                   'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
                                   'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
                                   'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename,
                            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                                sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
                             model_type,
                            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
                            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')

    # Dataset
    # return a waveform and a one-hot encoded target.
    # The training csv file filtered minor classes by a Dropping_threshold (10)
    train_csv = pd.read_csv("Germany_Birdcall_resampled_filtered.csv")
    classes_num = len(train_csv["gen"].unique())
    audio_path = "/mnt/GermanyBirdcall/Germany_Birdcall_resampled"
    # Split csv file training and test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    for train_idx, test_idx in splitter.split(X=train_csv, y=train_csv["gen"]):
        train_df = train_csv.loc[train_idx, :].reset_index(drop=True)
        test_df = train_csv.loc[test_idx, :].reset_index(drop=True)
    # dataset = WaveformDataset(df: pd.DataFrame, datadir: str)
    train_dataset = WaveformDataset(df=train_df, datadir=audio_path)
    test_dataset = WaveformDataset(df=test_df, datadir=audio_path)

    # Train sampler and Train loader
    num_workers = 10
    if balanced == 'balanced':
        train_sampler = BalancedSampler(
            df=train_df,
            batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=num_workers)

    # Evaluate sampler
    # eval_bal_sampler = BalancedSampler(df=test_df, batch_size=batch_size)
    # eval_bal_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset,
    #     batch_sampler=eval_bal_sampler,
    #     collate_fn=collate_fn,
    #     num_workers=num_workers,
    #     pin_memory=True)

    eval_test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers)

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    # Model Initialization
    Model = eval(model_type)  # model_type = "Transfer_Cnn14"
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                  classes_num, freeze_base)

    logging.info(args)
    if 'cuda' in str(device):
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    # Load pretrained model
    # CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"/"Cnn10_mAP=0.380.pth"/"Cnn6_mAP=0.343.pth"
    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)
        print('Load pretrained model successfully!')
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    if 'cuda' in device:
        model.to(device)

    # Loss
    loss_func = get_loss_func(loss_type)

    # Evaluator
    evaluator = Evaluator(model=model)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Training Loop
    train_bgn_time = time.time()
    time1 = time.time()
    iteration = 0
    best_mAP = 0
    path_best = ".\\best_model_mAP.pth"
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """

        # Evaluate
        if iteration % 2000 == 0 or (iteration == 0):
            train_fin_time = time.time()

            # bal_statistics = evaluator.evaluate(eval_bal_loader)
            test_statistics = evaluator.evaluate(eval_test_loader)

            # logging.info('Validate bal mAP: {:.3f}'.format(
            #     np.mean(bal_statistics['average_precision'])))

            logging.info('Validate test mAP: {:.3f}'.format(
                np.mean(test_statistics['average_precision'])))
            print('Validate test mAP: {:.3f}'.format(
                np.mean(test_statistics['average_precision'])))

            # statistics_container.append(iteration, bal_statistics, data_type='bal')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()
            
            # save best model
            if np.mean(test_statistics['average_precision'])) > best_mAP:
              best_mAP = np.mean(test_statistics['average_precision']))
              best_model_mAP = copy.deepcopy(model.state_dict())

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()  # reset after evaluation

        # Save model
        if iteration % 3000 == 0:
            checkpoint = {
                'iteration': iteration,
                'model': model.module.state_dict(),
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))

            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

        # Mixup lambda
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        # Forward
        model.train()

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'],
                                      batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'],
                                                    batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

        # Loss
        loss = loss_func(batch_output_dict, batch_target_dict)

        # Backward
        loss.backward()
        print(loss)

        optimizer.step()
        optimizer.zero_grad()

        if iteration % 100 == 0:
            print('--- Iteration: {}, train time: {:.3f} s / 100 iterations ---' \
                  .format(iteration, time.time() - time1))
            time1 = time.time()

        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1

    torch.save(best_model_mAP, path_best)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    # parser_train.add_argument('--cuda', action='store_true', default=True)
    # append from main.py
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--sample_rate', type=int, default=32000)
    parser_train.add_argument('--window_size', type=int, default=1024)
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=64)
    parser_train.add_argument('--fmin', type=int, default=50)
    parser_train.add_argument('--fmax', type=int, default=14000)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, default='clip_bce', choices=['clip_bce'])
    parser_train.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none', 'mixup'])
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--early_stop', type=int, default=300000)  # ~10000 train samples, 32 batch size, ~100 epochs
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
#     parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--freeze_base', default=False)
    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')
