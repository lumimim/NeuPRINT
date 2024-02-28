import numpy as np

def split_train_valid_test(arr, test_unique_ids_all_sessions, train_frac = 0.8):
    train_val_arr = np.setdiff1d(arr, test_unique_ids_all_sessions)
    test_arr = np.intersect1d(arr, test_unique_ids_all_sessions)
    unique_vals = np.unique(train_val_arr)
    unique_test_subset = np.unique(test_arr)
    num_unique = len(unique_vals)
    train_num = int(num_unique * train_frac)
    np.random.seed(0)
    train_indexes = np.random.choice(np.arange(0, unique_vals.shape[0]), train_num, replace = False)
    val_indexes = np.setdiff1d(np.arange(0, unique_vals.shape[0]), train_indexes)
    unique_train_subset = unique_vals[train_indexes]
    unique_val_subset = unique_vals[val_indexes]
    np.random.seed(0)
    train = [x for x in arr if x in unique_train_subset]
    valid = [x for x in arr if x in unique_val_subset]
    test = [x for x in arr if x in unique_test_subset]
    return train, valid, test

def extract_train_with_valid_test(arr, valid_test):
    train = [x for x in arr if x not in valid_test]
    return train