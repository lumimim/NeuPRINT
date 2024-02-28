import numpy as np
from bisect import bisect
from sklearn.neighbors import NearestNeighbors

def interpolate_nans(arr):
    nan_indices = np.isnan(arr)
    non_nan_arr = arr[~nan_indices]
    non_nan_indices = np.arange(len(arr))[~nan_indices]
    interp_arr = np.interp(np.arange(len(arr)), non_nan_indices, non_nan_arr)
    return interp_arr

def find_unique(arrays_list):
    unique_values = np.unique(np.concatenate(arrays_list))
    return unique_values

def count_nonan(arrays_list):
    nonan_count = np.count_nonzero(~np.isnan(np.concatenate(arrays_list)))
    return nonan_count

def replace_not_EI(lst):
    return ['IN' if x not in ['IN', 'EC'] else x for x in lst]

def change_types_to_subclasses(lst):
    output = []
    for x in lst:
        if x in ['IN', 'EC']:
            output.append('NaN')
        else:
            output.append(x.split('-')[0])
    return output

def find_neighbors_and_non_neighbors(points, D):
    """
    Finds the neighbor and non-neighbor indexes within distance D for each point in the input array, excluding the points themselves.

    Args:
    points (numpy.ndarray): A 2D array with shape (N, 2) representing the points.
    D (float): The maximum distance for a point to be considered a neighbor.

    Returns:
    Tuple[List[List[int]], List[List[int]]]: A tuple containing two lists of lists:
        1. The indexes of the neighbors within distance D for each point, excluding the points themselves.
        2. The indexes of the non-neighbors for each point.
    """
    # Initialize the NearestNeighbors object
    nbrs = NearestNeighbors(radius=D, algorithm='auto').fit(points)
    # Find the distances and indexes of the nearest neighbors within distance D
    neighbors = nbrs.radius_neighbors(points, return_distance=False)
    # Convert neighbors to lists of indexes and exclude each point from its own list of neighbors
    neighbors_indexes = []
    non_neighbors_indexes = []
    for i, neighbor_indexes in enumerate(neighbors):
        filtered_neighbor_indexes = [idx for idx in neighbor_indexes if idx != i]
        neighbors_indexes.append(filtered_neighbor_indexes)

        filtered_non_neighbor_indexes = [idx for idx in range(len(points)) if idx != i and idx not in filtered_neighbor_indexes]
        non_neighbors_indexes.append(filtered_non_neighbor_indexes)

    return neighbors_indexes, non_neighbors_indexes

def load_data_session(directory, date_exp, input_setting, behavior_normalization = True, session_normalization = False):
    gene_count = np.load(directory + date_exp + 'neuron.gene_count.npy')
    print('gene_count.shape:', gene_count.shape)
    UniqueID = np.load(directory + date_exp + 'neuron.UniqueID.npy')
    print('UniqueID.shape:', UniqueID.shape)

    with open(directory + date_exp + 'neuron.ttype.txt') as f:
        neuron_ttypes_raw  = f.readlines()
    neuron_ttypes = []
    for neuron_ttype in neuron_ttypes_raw:
        neuron_ttypes.append(neuron_ttype.strip())

    frame_states = np.load(directory + date_exp + input_setting + 'frame.states.npy')
    frame_times = np.load(directory + date_exp + input_setting + 'frame.times.npy')
    print('frame_states.shape:', frame_states.shape)

    frame_activity = np.load(directory + date_exp + input_setting + 'frame.neuralActivity.npy')
    frame_times = np.load(directory + date_exp + input_setting + 'frame.times.npy')
    print('frame_activity.shape:', frame_activity.shape)

    eye_size = np.load(directory + date_exp + input_setting + 'eye.size.npy')
    eye_times = np.load(directory + date_exp + input_setting + 'eye.times.npy')
    print('eye_size.shape:', eye_size.shape)

    if np.isnan(np.mean(eye_size)):
        eye_size = interpolate_nans(eye_size[:,0])[:,np.newaxis]

    times = frame_times.shape[0]
    eye_times_list = list(eye_times[:,0])
    eye_size_resize = np.zeros((times, 1))
    for t in range(times):
        index = bisect(eye_times_list, frame_times[t,0])
        if t == times - 1:
            eye_size_resize[t] = eye_size[index - 1, 0]
        else:
            eye_size_resize[t] = (eye_size[index - 1, 0] + eye_size[index, 0])/2

    running_speed = np.load(directory + date_exp + input_setting + 'running.speed.npy')
    running_times = np.load(directory + date_exp + input_setting + 'running.times.npy')
    print('running_speed.shape:', running_speed.shape)

    if np.isnan(np.mean(running_speed)):
        running_speed = interpolate_nans(running_speed[:,0])[:,np.newaxis]

    running_times_list = list(running_times[:,0])
    running_speed_resize = np.zeros((times, 1))
    for t in range(times):
        index = bisect(running_times_list, frame_times[t,0])
        if t == times - 1:
            running_speed_resize[t] = running_speed[index - 1, 0]
        else:
            running_speed_resize[t] = (running_speed[index - 1, 0] + running_speed[index, 0])/2

    # normalization
    if session_normalization:
        activity_mean = np.mean(frame_activity)
        activity_std = np.std(frame_activity)
    else:
        activity_mean = np.mean(frame_activity, axis = 0)
        activity_std = np.std(frame_activity, axis = 0)
    activity_norm = (frame_activity - activity_mean)/activity_std
    activity_norm = np.where(np.isnan(activity_norm), 0.0, activity_norm)

    if behavior_normalization:
        running_speed_resize = (running_speed_resize - np.mean(running_speed_resize, axis = 0))/np.std(running_speed_resize, axis = 0)
        eye_size_resize = (eye_size_resize - np.mean(eye_size_resize, axis = 0))/np.std(eye_size_resize, axis = 0)

    # activity population
    activity_population = {
        'running_speed': running_speed_resize,
        'eye_size': eye_size_resize,
        'frame_states': frame_states,
        'mean_activity': np.mean(activity_norm, axis = 1, keepdims = True),
        'std_activity': np.std(activity_norm, axis = 1, keepdims = True),
        }
    
    neuron_pos = np.load(directory + date_exp +'neuron.stackPosCorrected.npy')
    print('neuron_pos.shape:', neuron_pos.shape)

    return  activity_norm, activity_population, frame_times, UniqueID, neuron_ttypes, neuron_pos