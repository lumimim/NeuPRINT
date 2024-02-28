from torch.utils.data import Dataset
import torch
import numpy as np

# dataloader function: sampling batch over time dimension and neuron dimension
class Activity_All_Sessions_Dataset(Dataset):
    def __init__(self, data, class_data, record_unique_ids, device, train_val_dataset_ids, test_dataset_ids, num_batch, train_time_steps_frac, record_frac, window_len, batch_size, num_sample_neurons, population_keys, is_Training, is_posthoc, data_type = None):
        self.data = data
        self.class_data = class_data
        self.record_unique_ids = record_unique_ids
        self.device = device
        self.train_val_dataset_ids = train_val_dataset_ids
        self.num_dataset = len(self.data)
        self.num_batch = num_batch
        self.train_time_steps_frac = train_time_steps_frac
        self.record_frac = record_frac
        self.window_len = window_len
        self.batch_size = batch_size
        self.num_sample_neurons = num_sample_neurons
        self.population_keys = population_keys
        self.is_Training = is_Training
        self.is_posthoc = is_posthoc
        self.data_type = data_type
        
    def __len__(self):
        return self.num_batch
    
    def __getitem__(self, index):
        if self.data_type == 'train' or 'val':
            dataset_rand_id =  np.random.choice(np.asarray(self.train_val_dataset_ids))
        if self.data_type == 'test':
            dataset_rand_id =  np.random.choice(np.asarray(self.test_dataset_ids))
        activity_norm = self.data[dataset_rand_id]['activity_norm']
        activity_population = self.data[dataset_rand_id]['activity_population']
        new_unique_ids = self.data[dataset_rand_id]['new_unique_ids']
        activity_neighbor = self.data[dataset_rand_id]['activity_neighbor']
        session_time_steps = activity_norm.shape[0]

        overlap = np.intersect1d(new_unique_ids, self.record_unique_ids)
        not_overlap = np.setdiff1d(new_unique_ids, self.record_unique_ids)

        if not self.is_posthoc:
            exclude_neuron_unique_ids = np.asarray(list(self.class_data['EI_val_neuron_unique_ids']) + list(self.class_data['EI_test_neuron_unique_ids']))
            overlap_exclude = np.setdiff1d(overlap, exclude_neuron_unique_ids)
            not_overlap_exclude = np.setdiff1d(not_overlap, exclude_neuron_unique_ids)
            num_overlap = np.shape(overlap_exclude)[0]
            num_not_overlap = np.shape(not_overlap_exclude)[0]
            rand_neuron_overlap_indexed = np.random.choice(np.arange(0, num_overlap), int(self.record_frac * self.num_sample_neurons), replace=True)
            rand_neuron_not_overlap_indexed = np.random.choice(np.arange(0, num_not_overlap), self.num_sample_neurons - int(self.record_frac * self.num_sample_neurons), replace=True)
            
            select_overlap_ids = overlap_exclude[rand_neuron_overlap_indexed]
            select_not_overlap_ids = not_overlap_exclude[rand_neuron_not_overlap_indexed]
            select_neuron_unique_ids = np.concatenate((select_overlap_ids, select_not_overlap_ids), axis = 0)
            
            select_overlap_indexed = []
            select_not_overlap_indexed = []
            for id_item in list(select_overlap_ids):
                select_overlap_indexed.append(np.where(new_unique_ids == id_item)[0])
            for id_item in list(select_not_overlap_ids):
                select_not_overlap_indexed.append(np.where(new_unique_ids == id_item)[0])
            select_neuron_indexed = np.concatenate(select_overlap_indexed + select_not_overlap_indexed)
        else:
            if self.data_type == 'train':
                neuron_unique_ids = self.class_data['EI_train_neuron_unique_ids']
            elif self.data_type == 'val':
                neuron_unique_ids = self.class_data['EI_val_neuron_unique_ids']
            elif self.data_type == 'test':
                neuron_unique_ids = self.class_data['EI_test_neuron_unique_ids']
            else:
                return KeyError
            overlap = np.intersect1d(new_unique_ids, neuron_unique_ids)
            num_overlap = np.shape(overlap)[0]
            rand_neuron_overlap_indexed = np.random.choice(np.arange(0, num_overlap), self.num_sample_neurons, replace=True)
            select_overlap_ids = overlap[rand_neuron_overlap_indexed]
            select_neuron_unique_ids = np.squeeze(np.asarray(select_overlap_ids))
            select_overlap_indexed = []
            for id_item in list(select_overlap_ids):
                select_overlap_indexed.append(np.where(new_unique_ids == id_item)[0])
            select_neuron_indexed = np.squeeze(np.asarray(select_overlap_indexed))
        
        num_train_steps = int(session_time_steps * self.train_time_steps_frac)
        # population activity
        population_feature_dim = len(self.population_keys) # p
        population_all_features =  np.zeros((session_time_steps, population_feature_dim)) # T x p
        for i, population_key in enumerate(self.population_keys):
            population_all_features[:, i] = activity_population[population_key][:,0]
        if self.is_Training:
            rand_time_indexes = np.random.choice(np.arange(0, num_train_steps - self.window_len), self.batch_size, replace = False)
        else:
            rand_time_indexes = np.random.choice(np.arange(num_train_steps, session_time_steps - self.window_len), self.batch_size, replace=False)

        F_trial = np.zeros((self.batch_size, self.num_sample_neurons, self.window_len))
        F_population_trial = np.zeros((self.batch_size, population_feature_dim, self.window_len)) # b x p # t
        F_neighbor_trial = np.zeros((self.batch_size, self.num_sample_neurons, 2, self.window_len))
        # holdout time steps during reconstruction
        for i in range(self.batch_size): 
            F_trial[i] = activity_norm[rand_time_indexes.astype(int)[i]:rand_time_indexes.astype(int)[i]+self.window_len, select_neuron_indexed].T
            # permutation invariant
            F_population_trial[i] = population_all_features[rand_time_indexes.astype(int)[i]: rand_time_indexes.astype(int)[i] + self.window_len, :].T
            F_neighbor_trial[i] = np.transpose(activity_neighbor[rand_time_indexes.astype(int)[i]: rand_time_indexes.astype(int)[i] + self.window_len, select_neuron_indexed, :],(1,2,0)) 

        x_ts = torch.Tensor(F_trial).to(self.device)
        pop_x_ts = torch.Tensor(F_population_trial).to(self.device)
        neigh_x_ts = torch.Tensor(F_neighbor_trial).to(self.device)
        neuron_id_ts = torch.Tensor(select_neuron_unique_ids).to(self.device)
    
        return x_ts, pop_x_ts, neigh_x_ts, neuron_id_ts