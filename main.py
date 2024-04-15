import sys, os, numpy as np, csv
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from multiprocessing import Pool
import pandas as pd
import logging
import torch
import gc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False)
torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
from data_preprocess import load_data_session, find_unique, count_nonan, replace_not_EI, change_types_to_subclasses, find_neighbors_and_non_neighbors
from neuprint import eval_single_neuron_time_invariant_permutation_invariant_recon_classfier
from label_generation import record_id_subclass_label_generation
import argparse
import datetime
from pathlib import Path
import pdb

import subprocess as sp
from threading import Thread , Timer
import sched, time
now = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-tag",
        required=True,
        help="name tag of experiment",
    )
    return parser

parser = get_parser()
args = parser.parse_args()
args_dict = vars(args)
exp_tag = args_dict['exp_tag']

directory = '../../../data/Bugeon/'
img_picture = np.load(directory + 'img.picture.npy')
print(img_picture.shape)
with open(directory + 'ttypes.names.txt') as f:
    ttypes_raw  = f.readlines()
ttypes = []
for ttype in ttypes_raw:
    ttypes.append(ttype.strip())
print(ttypes)
with open(directory + 'genes.names.txt') as f:
    genes_raw  = f.readlines()
genes = []
for gene in genes_raw:
    genes.append(gene.strip())
print(genes)

data_all_sessions = []

date_exp_all  = ['SB025/2019-10-07/', 
                 'SB025/2019-10-04/', 
                 'SB025/2019-10-08/',
                 'SB025/2019-10-09/',
                 'SB025/2019-10-23/',
                 'SB025/2019-10-24/',
                 'SB026/2019-10-11/',
                 'SB026/2019-10-14/',
                 'SB026/2019-10-16/',
                 'SB028/2019-11-06/',
                 'SB028/2019-11-07/',
                 'SB028/2019-11-08/',
                 'SB028/2019-11-12/',
                 'SB028/2019-11-13/',
                 'SB030/2020-01-08/',
                 'SB030/2020-01-10/',
                 'SB030/2020-01-28/']
train_val_dataset_ids = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
test_dataset_ids = [5, 8, 13, 16]

for i in range(len(date_exp_all)):
    date_exp = date_exp_all[i]
    # input_setting = 'Drifting Gratings/01/'
    # input_setting = 'Natural Scenes/01/'
    # input_setting = 'Natural Scenes/02/'
    input_setting = 'Blank/01/'
    activity_norm, activity_population, frame_times, unique_ids, neuron_types, neuron_pos = load_data_session(directory, date_exp, input_setting, session_normalization = False)
    data_all_sessions.append({'activity_norm': activity_norm,
                            'activity_population': activity_population, 
                            'frame_times': frame_times, 
                            'unique_ids': unique_ids, 
                            'neuron_types': neuron_types,
                             'neuron_pos': neuron_pos})

print('number of session per animal:', len(data_all_sessions))
print(data_all_sessions[0].keys())
unique_inds_all = find_unique([data_all_sessions[i]['unique_ids'] for i in range(len(data_all_sessions))])[:-1]
print(unique_inds_all.shape)
print('count_nonan:', count_nonan([data_all_sessions[i]['unique_ids'] for i in range(len(data_all_sessions))]))

print('count_unique_ids:',unique_inds_all.shape[0])
new_id_start = unique_inds_all.shape[0]
print('new_id_start:',new_id_start)
for dataset_id in range(len(data_all_sessions)):
    new_unique_ids = data_all_sessions[dataset_id]['unique_ids'][:,0].copy()
    new_EI_classes = np.asarray(replace_not_EI(data_all_sessions[dataset_id]['neuron_types']))
    new_subclasses = np.asarray(change_types_to_subclasses(data_all_sessions[dataset_id]['neuron_types']))
    not_record_neuron_session_indexes = np.where(np.isnan(data_all_sessions[dataset_id]['unique_ids'][:,0]))[0]
    record_neuron_session_indexes = np.where(~np.isnan(data_all_sessions[dataset_id]['unique_ids'][:,0]))[0]
    not_record_count = not_record_neuron_session_indexes.shape[0]
    new_unique_ids[not_record_neuron_session_indexes] = new_id_start + np.arange(not_record_count)
    data_all_sessions[dataset_id]['new_unique_ids'] = new_unique_ids
    data_all_sessions[dataset_id]['new_EI_classes'] = new_EI_classes
    data_all_sessions[dataset_id]['new_subclasses'] = new_subclasses
    data_all_sessions[dataset_id]['record_unique_ids'] = new_unique_ids[record_neuron_session_indexes]
    data_all_sessions[dataset_id]['record_subclasses'] = new_subclasses[record_neuron_session_indexes]
    data_all_sessions[dataset_id]['record_activity_norm'] = data_all_sessions[dataset_id]['activity_norm'][:,record_neuron_session_indexes]
    new_id_start = new_id_start + not_record_count

print(data_all_sessions[0].keys())
for dataset_id in range(len(data_all_sessions)):
    print('dataset: time_steps x num_neuron', data_all_sessions[dataset_id]['activity_norm'].shape)

class_all_sessions, all_sessions_neuron_count = record_id_subclass_label_generation(data_all_sessions, test_dataset_ids)

neigbors_dist = 65
for dataset_id in range(len(data_all_sessions)):
    activity_norm = data_all_sessions[dataset_id]['activity_norm']
    neuron_pos = data_all_sessions[dataset_id]['neuron_pos']
    num_neuron = activity_norm.shape[1]
    neuron_neighbors, neuron_non_neighbors = find_neighbors_and_non_neighbors(neuron_pos, neigbors_dist)
    F_neighbor = np.zeros((activity_norm.shape[0], num_neuron, 2))
    for j in range(num_neuron):
        if len(np.asarray(neuron_neighbors[j])) != 0:
            F_neighbor[:, j, 0] = np.mean(activity_norm[:, np.asarray(neuron_neighbors[j])], axis = 1)
            F_neighbor[:, j, 1] = np.std(activity_norm[:, np.asarray(neuron_neighbors[j])], axis = 1)
    data_all_sessions[dataset_id]['activity_neighbor'] = F_neighbor

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

hparams_dict = {
          'recon_model_type': 'transformer',
          'classify_model_type': 'mlp',
          'output_distribution': 'mse',
          'use_population': True,
          'use_neighbor': True,
          'return_last': False,
          'all_sessions_neuron_count': int(all_sessions_neuron_count),
          'num_batch': 100,
          'batch_size': 2,
          'record_frac': 0.9,
          'post_hoc_eval_epoch_freq': 50,
          'num_sample_neurons': 512,
          'num_epochs': 400,
          'num_post_hoc_epochs': 5000, # default 5000
          'num_z_optimize_epochs': 100,
          'z_optimize_lr': 1e-3,
          'hidden_dim': 5, # default 32 ### to be divisible by nhead after concat with population features
          'post_hoc_hidden_dim': 2048, # default = hidden_dim
          'embedding_dim': 64, # 8 # default 8
          'layer_dim': 1, # default 1
          'window_len': 200, # default 200 ###
          'full_context': False, # default True ###
          'full_causal': False, # default False ###
          'context_backward': 1, # default -1 ###
          'context_forward': 0, # default -1 ###
          'attend_to_self': True, # default True ###
          'ckpt_path': None, # default None ###
          'use_ckpt_hparams': True, # default True,
          'post_hoc_only': False, # default False,
          'use_weighted_ce_loss': False # default False
}

recon_model = hparams_dict['recon_model_type']
layer_dim = hparams_dict['layer_dim']
embedding_dim = hparams_dict['embedding_dim']
output_distribution = hparams_dict['output_distribution']
num_z_optimize_epochs = hparams_dict['num_z_optimize_epochs']
post_hoc_eval_epoch_freq = hparams_dict['post_hoc_eval_epoch_freq']
exp_name = f'train_val_mice_split_blank_data_F_z_{recon_model}_loss_{output_distribution}_layer_{layer_dim}_embedding_{embedding_dim}_z_epoch_{num_z_optimize_epochs}_eval_freq_{post_hoc_eval_epoch_freq}_population_neighbor_ckpt_large_mlp_all_mouses'

tensorboard_dir = '../../../tensorboard/'
ckpt_dir = '../../../checkpoints/' + exp_name + '/'
writer = SummaryWriter(os.path.join(tensorboard_dir, f'{now}_{exp_tag}'))

if not Path(ckpt_dir).exists():
    os.mkdir(ckpt_dir)

logging.basicConfig(filename = '../../../experiments/logs/'+exp_name+'.log',level=logging.DEBUG,format='%(message)s')
logging.debug('hyperparams log: %s', hparams_dict)
gc.collect()
torch.cuda.empty_cache()
outputs = eval_single_neuron_time_invariant_permutation_invariant_recon_classfier(
          data_all_sessions = data_all_sessions,
          train_val_dataset_ids = train_val_dataset_ids,
          test_dataset_ids = test_dataset_ids,
          logging = logging,
          writer = writer,
          **hparams_dict
          )
print(outputs['metrics']['EI_train_mlp'])
print(outputs['metrics']['EI_valid_mlp'])
print(outputs['metrics']['EI_test_mlp'])
print(outputs['metrics']['subclass_train_mlp'])
print(outputs['metrics']['subclass_valid_mlp'])
print(outputs['metrics']['subclass_test_mlp'])
metrics_dict = {'best epoch posthoc EI train acc': outputs['metrics']['EI_train_mlp'],
                'best posthoc EI valid acc': outputs['metrics']['EI_valid_mlp'],
                'best epoch posthoc EI test acc': outputs['metrics']['EI_test_mlp'],
                'best epoch posthoc subclass train acc': outputs['metrics']['subclass_train_mlp'],
                'best posthoc subclass valid acc': outputs['metrics']['subclass_valid_mlp'],
                'best epoch posthoc subclass test acc': outputs['metrics']['subclass_test_mlp']}
writer.add_hparams(hparams_dict, metrics_dict)
logging.debug('best posthoc acc: %s', outputs['metrics'])
plt.close('all')
writer.close()
