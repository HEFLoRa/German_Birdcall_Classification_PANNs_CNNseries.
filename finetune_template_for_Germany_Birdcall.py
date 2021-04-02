import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))
print("system path:")
print(sys.path)
# import functions
from utils.utilities import (create_folder, get_filename, create_logging, Mixup, StatisticsContainer)
from pytorch.models import *  # import all models
from pytorch.pytorch_utils import (move_data_to_device, do_mixup)
from pytorch.evaluate import Evaluator
from pytorch.losses import get_loss_func
# Germany Birdcall
from pytorch.Germany_Birdcall_dataset_preprocessing import WaveformDataset, BalancedSampler, collate_fn
from sklearn.model_selection import StratifiedShuffleSplit
from pytorch.Transfer_models import *  # import all transfer_models
import numpy as np
import pandas as pd
import argparse
import time
import logging
import copy
import torch
import torch.optim as optim
import torch.utils.data
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)


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
    # for fine-tune models
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base_num = args.freeze_base_num

    pretrain = True if pretrained_checkpoint_path else False

    # Define Saving Paths
    best_model_path = os.path.join(workspace, 'best_model', filename,
                                   'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'
                                   .format(sample_rate, window_size, hop_size, mel_bins, fmin, fmax), model_type,
                                   'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
                                   'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
                                   )
    create_folder(os.path.dirname(best_model_path))

    statistics_path = os.path.join(workspace, 'statistics', filename,
                                   'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'
                                   .format(sample_rate, window_size, hop_size, mel_bins, fmin, fmax), model_type,
                                   'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
                                   'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
                                   'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename,
                            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'
                            .format(sample_rate, window_size, hop_size, mel_bins, fmin, fmax), model_type,
                            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
                            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_logging(logs_dir, filemode='w')

    # Dataset
    # return a waveform and a one-hot encoded target.
    # The training csv file filtered minor classes by a Dropping_threshold (10)
    train_csv = pd.read_csv("/mnt/Germany_Birdcall/German-Birdcall/Germany_Birdcall_resampled_filtered.csv")
    classes_num = len(train_csv["gen"].unique())
    audio_path = "/mnt/Germany_Birdcall/Germany_Birdcall_resampled"
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

    eval_test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers)

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    # Model Initialization
    transfer_model = eval(model_type)  # model_type = "Transfer_Cnn14"
    model = transfer_model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                  classes_num, freeze_base_num)

    logging.info(args)

    # Load pretrained model
    # CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"/"Cnn10_mAP=0.380.pth"/"Cnn6_mAP=0.343.pth"
    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)
        print('Load pretrained model successfully!')
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'cuda' in device:
        model.to(device)
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    # Loss
    loss_func = get_loss_func(loss_type)

    # Evaluator : return mAP and Auc value
    evaluator = Evaluator(model=model)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Training Loop
    time_initial = time.time()
    train_bgn_time = time.time()
    time1 = time.time()
    iteration = 0
    best_mAP = 0
    # store validation results with pd.dataframe
    validation_results = pd.DataFrame(columns=["iteration","mAP","Auc"])
    i = 0
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """

        # Evaluate
        if iteration % 200 == 0 or (iteration == 0):
            train_fin_time = time.time()

            test_statistics = evaluator.evaluate(eval_test_loader)
            current_mAP = np.mean(test_statistics['average_precision'])
            current_auc = np.mean(test_statistics['auc'])
            logging.info('Validate test mAP: {:.3f}'.format(current_mAP))
            logging.info('Validate test Auc: {:.3f}'.format(current_auc))
            validation_results.loc[i] = [iteration, current_mAP, current_auc]
            i += 1

            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()
            
            # copy best model
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                best_model = copy.deepcopy(model.state_dict())

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(iteration, train_time, validate_time))
            logging.info('------------------------------------')

            train_bgn_time = time.time()  # reset after evaluation

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

        if iteration % 200 == 0:
            print('--- Iteration: {}, train time: {:.3f} s / 200 iterations ---' \
                  .format(iteration, time.time() - time1))
            time1 = time.time()

        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1

    # Save model
    best_model_path = "best_"+model_type+balanced+augmentation+"freeze"\
                      + str(freeze_base_num)+"_mAP={.3f}".format(best_mAP)
    torch.save(best_model, best_model_path+".pth")

    # Save validation results
    validation_results_path = "validation_results"+model_type+balanced\
                              + augmentation+"freeze"+str(freeze_base_num)\
                              + "_mAP={.3f}".format(best_mAP)
    validation_results.to_csv(validation_results_path+'.csv', index=False)

    time_end = time.time()
    time_cost = time_end - time_initial
    print("The whole training process takes: {.3f} s".format(time_cost))


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
    parser_train.add_argument('--early_stop', type=int, default=20000)  # early_stop * batch_size / num_trn_samples = epoch
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
#     parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--freeze_base_num', type=int, default=0)
    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')
