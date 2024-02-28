import numpy as np
from plot_func import plot_histogram
from data_split import split_train_valid_test, extract_train_with_valid_test

def record_id_subclass_label_generation(data_all_sessions, test_dataset_ids, train_neurons_frac = 0.8, val_neurons_frac = 0.1):
    record_neuron_subclasses_all_sessions = []
    record_unique_neuron_ids_all_sessions = []
    neuron_EI_class_all_sessions = []
    neuron_unique_ids_all_sessions = []

    test_neuron_unique_ids_all_sessions_list = []
    
    for dataset_id in range(len(data_all_sessions)):
        neuron_EI_class_all_sessions += list(data_all_sessions[dataset_id]['new_EI_classes'])
        neuron_unique_ids_all_sessions += list(data_all_sessions[dataset_id]['new_unique_ids'])
        record_unique_neuron_ids_all_sessions += list(data_all_sessions[dataset_id]['record_unique_ids'])
        record_neuron_subclasses_all_sessions += list(data_all_sessions[dataset_id]['record_subclasses'])

        if dataset_id in test_dataset_ids:
            test_neuron_unique_ids_all_sessions_list += list(data_all_sessions[dataset_id]['new_unique_ids'])

    all_sessions_neuron_count = max(neuron_unique_ids_all_sessions) + 1
    print('all_sessions_neuron_acount:', all_sessions_neuron_count)
    plot_histogram(record_neuron_subclasses_all_sessions)
    plot_histogram(neuron_EI_class_all_sessions)
    
    EC_class_index = np.where(np.asarray(neuron_EI_class_all_sessions) == 'EC')[0]
    IN_class_index = np.where(np.asarray(neuron_EI_class_all_sessions) == 'IN')[0]

    pv_class_index = np.where(np.asarray(record_neuron_subclasses_all_sessions) == 'Pvalb')[0]
    la_class_index = np.where(np.asarray(record_neuron_subclasses_all_sessions) == 'Lamp5')[0]
    sn_class_index =  np.where(np.asarray(record_neuron_subclasses_all_sessions) == 'Sncg')[0]
    ss_class_index =  np.where(np.asarray(record_neuron_subclasses_all_sessions) == 'Sst')[0]
    vi_class_index = np.where(np.asarray(record_neuron_subclasses_all_sessions) == 'Vip')[0]

    EC_class = np.asarray(neuron_unique_ids_all_sessions)[EC_class_index]
    IN_class = np.asarray(neuron_unique_ids_all_sessions)[IN_class_index]
    
    # mapping back to original neuron ID space
    pv_class = np.asarray(record_unique_neuron_ids_all_sessions)[pv_class_index]
    la_class = np.asarray(record_unique_neuron_ids_all_sessions)[la_class_index]
    sn_class = np.asarray(record_unique_neuron_ids_all_sessions)[sn_class_index]
    ss_class = np.asarray(record_unique_neuron_ids_all_sessions)[ss_class_index]
    vi_class = np.asarray(record_unique_neuron_ids_all_sessions)[vi_class_index]

    test_neuron_unique_ids_all_sessions = np.asarray(test_neuron_unique_ids_all_sessions_list)
    pv_class_train, pv_class_val, pv_class_test = split_train_valid_test(pv_class, test_neuron_unique_ids_all_sessions, train_neurons_frac)
    la_class_train, la_class_val, la_class_test = split_train_valid_test(la_class, test_neuron_unique_ids_all_sessions, train_neurons_frac)
    sn_class_train, sn_class_val, sn_class_test = split_train_valid_test(sn_class, test_neuron_unique_ids_all_sessions, train_neurons_frac)
    ss_class_train, ss_class_val, ss_class_test = split_train_valid_test(ss_class, test_neuron_unique_ids_all_sessions, train_neurons_frac)
    vi_class_train, vi_class_val, vi_class_test  = split_train_valid_test(vi_class, test_neuron_unique_ids_all_sessions, train_neurons_frac)
    
    train_neuron_indexes = pv_class_train + la_class_train + sn_class_train + ss_class_train + vi_class_train
    val_neuron_indexes = pv_class_val + la_class_val + sn_class_val + ss_class_val + vi_class_val
    test_neuron_indexes = pv_class_test + la_class_test + sn_class_test + ss_class_test + vi_class_test

    IN_class_train, IN_class_val, IN_class_test = split_train_valid_test(IN_class, test_neuron_unique_ids_all_sessions, train_neurons_frac)
    EC_class_train, EC_class_val, EC_class_test = split_train_valid_test(EC_class, test_neuron_unique_ids_all_sessions, train_neurons_frac)

    EI_train_neuron_indexes = EC_class_train + IN_class_train
    EI_val_neuron_indexes = EC_class_val + IN_class_val
    
    EI_test_neuron_indexes = EC_class_test + IN_class_test

    class_all_sessions = {
        'record_subclass_all_sessions': np.asarray(record_neuron_subclasses_all_sessions),
        'record_unique_ids_all_sessions':np.asarray(record_unique_neuron_ids_all_sessions),
        'record_train_neuron_unique_ids':np.asarray(train_neuron_indexes),
        'record_val_neuron_unique_ids':np.asarray(val_neuron_indexes),
        'record_test_neuron_unique_ids':np.asarray(test_neuron_indexes),
        'neuron_EI_class_all_sessions': np.asarray(neuron_EI_class_all_sessions),
        'neuron_unique_ids_all_sessions': np.asarray(neuron_unique_ids_all_sessions),
        'EI_train_neuron_unique_ids': np.asarray(EI_train_neuron_indexes),
        'EI_val_neuron_unique_ids': np.asarray(EI_val_neuron_indexes),
        'EI_test_neuron_unique_ids': np.asarray(EI_test_neuron_indexes)
    }

    return class_all_sessions, all_sessions_neuron_count