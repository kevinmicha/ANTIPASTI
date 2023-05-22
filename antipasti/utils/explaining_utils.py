import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from copy import deepcopy
from matplotlib.colors import CenteredNorm

def get_maps_of_interest(preprocessed_data, learnt_filter, affinity_thr=-8):
    r"""Post-processes both raw data and results to obtain maps of interest.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    learnt_filter: numpy.ndarray
        Filters that express the learnt features during training.
    affinity_thr: float
        Affinity value separating antibodies considered to have high affinity from those considered to have low affinity.

    Returns
    -------
    mean_learnt: numpy.ndarray
        A resized version of ``learnt_filter`` to match the shape of the input normal mode correlation maps.
    mean_image: numpy.ndarray
        The mean of all the input normal mode correlation maps.
    mean_diff_image: numpy.ndarray
        Map resulting from the subtraction of the mean of the high affinity correlation maps and the mean of the low affinity correlation maps.

    """
    high_aff = []
    low_aff = []
    train_x = preprocessed_data.train_x
    train_y = preprocessed_data.train_y
    input_shape = train_x.shape[-1]

    for i in range(train_y.shape[0]):
        if train_y[i] < affinity_thr:
            high_aff.append(train_x[i])
        elif train_y[i] > affinity_thr:
            low_aff.append(train_x[i])

    # Obtaining the maps
    mean_learnt = cv2.resize(learnt_filter, dsize=(input_shape, input_shape))
    mean_image = np.mean(train_x, axis=0).reshape(input_shape, input_shape)
    mean_diff_image = np.mean(high_aff, axis=0) - np.mean(low_aff, axis=0)

    return mean_learnt, mean_image, mean_diff_image

def get_epsilon(preprocessed_data, model, mean_diff_image):
    r"""Returns a map ``epsilon`` (ϵ) such that the predicted affinity of x + ϵ is always greater than that of x.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    mean_diff_image: numpy.ndarray
        Map resulting from the subtraction of the mean of the high affinity correlation maps and the mean of the low affinity correlation maps.

    """
    high_aff = []
    low_aff = []
    train_x = preprocessed_data.train_x
    input_shape = preprocessed_data.train_x.shape[-1]
    n_filters = model.n_filters

    each_img_enl = np.zeros((train_x.shape[0], input_shape, input_shape))
    size_le = int(np.sqrt(model.fc1.weight.data.numpy().shape[-1] / n_filters))

    for j in range(train_x.shape[0]):
        inter_filter_item = model(torch.from_numpy(train_x[j].reshape(1, 1, input_shape, input_shape).astype(np.float32)))[1].detach().numpy()
        for i in range(n_filters):
            each_img_enl[j] += cv2.resize(np.multiply(inter_filter_item[0,i], model.fc1.weight.data.numpy().reshape(n_filters, size_le**2)[i].reshape(size_le, size_le)), dsize=(input_shape, input_shape))
        high_aff.append(np.multiply(-np.clip(each_img_enl[j], a_min=-np.inf, a_max=0), train_x[j]))
        low_aff.append(np.multiply(np.clip(each_img_enl[j], a_min=0, a_max=np.inf), train_x[j]))

    true_filter = deepcopy(np.mean(high_aff, axis=0) - np.mean(low_aff, axis=0))
    epsilon = np.multiply(np.abs(np.sign(mean_diff_image[0])), true_filter)

    return epsilon

def plot_map_with_regions(preprocessed_data, map, title='Normal mode correlation map'):
    r"""Maps the residues to the antibody regions and plots the normal mode correlation map.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    map: numpy.ndarray
        A normal mode correlation map.
    title: str
        The image title.

    """
    # Font sizes
    title_size = 20
    font_size = 14

    # Defining the region boundaries
    mrlh = preprocessed_data.max_res_list_h
    mrll = preprocessed_data.max_res_list_l

    subgroup_boundaries_h = [mrlh.index('26'), mrlh.index('33'), mrlh.index('39'), mrlh.index('45'), mrlh.index('51'),
                        mrlh.index('52'), mrlh.index('57'), mrlh.index('61'), mrlh.index('67'), mrlh.index('72'),
                        mrlh.index('75'), mrlh.index('82'), mrlh.index('84'), mrlh.index('87'), mrlh.index('89'),
                        mrlh.index('95'), mrlh.index('102')+1]
    subgroup_boundaries_l = [mrll.index('24'), mrll.index('34'), mrll.index('38'), mrll.index('44'), mrll.index('48'),
                        mrll.index('50'), mrll.index('56'), mrll.index('62'), mrll.index('65'), mrll.index('71'),
                        mrll.index('75'), mrll.index('85'), mrll.index('89'), mrll.index('97')+1]
    labels_h = ['CDR1', '\u03B211', '', '\u03B212', '', 'CDR2', '\u03B213', '', '\u03B221', '', '\u03B222',
            '', '\u03B1', '', '\u03B214', 'CDR3', '']
    labels_l = ['CDR1', '\u03B211', '', '\u03B212', '', 'CDR2', '', '\u03B221', '', '\u03B222',
            '', '\u03B213', 'CDR3']

    subgroup_boundaries = subgroup_boundaries_h + [x+mrlh.index('102')+1 for x in subgroup_boundaries_l]
    labels = labels_h + labels_l
    fig = plt.figure(figsize=(20, 20))

    # Plotting the normal mode correlation map
    plt.imshow(map, origin='lower', cmap='seismic', norm=CenteredNorm())
    plt.colorbar(fraction=0.045)

    for i in range(len(subgroup_boundaries) - 1):
        start_index = subgroup_boundaries[i]
        end_index = subgroup_boundaries[i+1]
        label_position = (start_index + end_index) / 2 - 0.5
        
        # Choosing the colours
        if labels[i].startswith('CDR'):
            c = 'orange'
        elif labels[i].startswith('\u03B2'): # Beta
            c = 'green'
        elif labels[i].startswith('\u03B1'): # Alpha
            c = 'red'
        else:
            c = 'white'
            
            
        # Adding rectangles for the regions
        rect = plt.Rectangle((start_index - 0.5, -6.5), end_index - start_index, 6,
                            edgecolor='black', facecolor=c, alpha=0.6)
        plt.gca().add_patch(rect)

        rect = plt.Rectangle((-12.5, start_index - 0.5), 12, end_index - start_index,
                            edgecolor='black', facecolor=c, alpha=0.6)
        plt.gca().add_patch(rect)

        # Add labels for the regions on the y-axis
        plt.text(-6, label_position-0.25, labels[i], ha='center', va='center', color='black', size=10)
        # Add labels for the regions on the x-axis
        plt.text(label_position, -3.5, labels[i], ha='center', va='center', color='black', size=8)

    # Adding rectangles for the chains
    rect = plt.Rectangle((-0.5, -10.5), subgroup_boundaries_h[-1], 4, edgecolor='black', facecolor='white')
    plt.gca().add_patch(rect)
    rect = plt.Rectangle((subgroup_boundaries_h[-1]-0.5, -10.5), subgroup_boundaries_l[-1], 4, edgecolor='black', facecolor='white')
    plt.gca().add_patch(rect)
    rect = plt.Rectangle((-16.5, -0.5), 4, subgroup_boundaries_h[-1], edgecolor='black', facecolor='white')
    plt.gca().add_patch(rect)
    rect = plt.Rectangle((-16.5, subgroup_boundaries_h[-1]-0.5), 4, subgroup_boundaries_l[-1], edgecolor='black', facecolor='white')
    plt.gca().add_patch(rect)

    # Adding labels for the chains on the y-axis
    plt.text(-14, subgroup_boundaries_h[-1]/2, 'Heavy chain', ha='center', va='center', color='black', rotation='vertical', size=14)
    plt.text(-14, subgroup_boundaries_h[-1]+subgroup_boundaries_l[-1]/2, 'Light chain', ha='center', va='center', color='black', rotation='vertical', size=14)
    # Adding labels for the chains on the x-axis
    plt.text(subgroup_boundaries_h[-1]/2, -8.75, 'Heavy chain', ha='center', va='center', color='black', size=14)
    plt.text(subgroup_boundaries_h[-1]+subgroup_boundaries_l[-1]/2, -8.75, 'Light chain', ha='center', va='center', color='black', size=14)

    # Adjusting the axis limits and labels
    plt.xlim(-16.5, 214.5)
    plt.ylim(-10.5, 214.5)
    plt.xticks([])
    plt.yticks([])

    # Adding labels and title
    plt.xlabel('Residues', size=font_size)
    plt.ylabel('Residues', size=font_size)
    plt.title(title, size=title_size)

    plt.show()

def compute_change_in_kd(preprocessed_data, model, weights, coord, maps):
    r"""Prints the percentage change in Kd when adding epsilon.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    weights: numpy.ndarray
        The weights given to each antibody region.
    coord: numpy.ndarray
        The coordinates of the antibody regions.
    maps: list
        The maps of the antibody regions coming from ``epsilon``.

    """
    input_shape = preprocessed_data.train_x.shape[-1]

    # Adding epsilon
    ideal = deepcopy(preprocessed_data.test_x)
    for i in range(len(weights)):
        temp = deepcopy(np.pad(maps[i], ((0, 0), (coord[i][0], ideal.shape[-1]-coord[i][-1]-1))))
        ideal += weights[i] * (temp + np.transpose(temp)) / 2

    # Comparing the new Kd w.r.t the original one
    prediction = 10**model(torch.from_numpy(preprocessed_data.test_x.reshape(1, 1, input_shape, input_shape).astype(np.float32)))[0].detach().numpy()
    new_prediction = 10**model(torch.from_numpy(ideal.reshape(1, 1, input_shape, input_shape).astype(np.float32)))[0].detach().numpy()
    per_change = ((prediction - new_prediction) / prediction * 100)[0][0]

    print('Without adding epsilon, Kd = ' + str(prediction[0,0]))
    print('After adding epsilon, Kd = ' + str(new_prediction[0,0]))
    print('Thus, Kd is smaller by', per_change, '%')


def map_residues_to_regions(preprocessed_data, epsilon):
    r"""Maps the residues to the antibody regions.

    Parameters
    ----------
    preprocessed_data: antipasti.model.model.Preprocessing
        The ``Preprocessing`` class.
    epsilon: numpy.ndarray
        A map ``epsilon`` (ϵ) such that the predicted affinity of x + ϵ is always greater than that of x.

    Returns
    -------
    coord: numpy.ndarray
        The coordinates of the antibody regions.
    maps: list
        The maps of the antibody regions coming from ``epsilon``.

    """
    max_res_list_h = preprocessed_data.max_res_list_h
    max_res_list_l = preprocessed_data.max_res_list_l

    cdr1_coord_h = range(max_res_list_h.index('26'), max_res_list_h.index('33'))
    cdr2_coord_h = range(max_res_list_h.index('52'), max_res_list_h.index('57'))
    cdr3_coord_h = range(max_res_list_h.index('95'), max_res_list_h.index('102')+1)
    beta11_coord_h = range(max_res_list_h.index('33'), max_res_list_h.index('39'))
    beta12_coord_h = range(max_res_list_h.index('45'), max_res_list_h.index('51'))
    beta13_coord_h = range(max_res_list_h.index('57'), max_res_list_h.index('61'))
    beta14_coord_h = range(max_res_list_h.index('89'), max_res_list_h.index('95'))
    beta21_coord_h = range(max_res_list_h.index('67'), max_res_list_h.index('72'))
    beta22_coord_h = range(max_res_list_h.index('75'), max_res_list_h.index('82'))
    alpha_coord_h = range(max_res_list_h.index('84'), max_res_list_h.index('87'))

    cdr1_coord_l = range(max_res_list_l.index('24'), max_res_list_l.index('34'))
    cdr2_coord_l = range(max_res_list_l.index('50'), max_res_list_l.index('56'))
    cdr3_coord_l = range(max_res_list_l.index('89'), max_res_list_l.index('97')+1)
    beta11_coord_l = range(max_res_list_l.index('34'), max_res_list_l.index('38'))
    beta12_coord_l = range(max_res_list_l.index('44'), max_res_list_l.index('48'))
    beta13_coord_l = range(max_res_list_l.index('85'), max_res_list_l.index('89'))
    beta21_coord_l = range(max_res_list_l.index('62'), max_res_list_l.index('65'))
    beta22_coord_l = range(max_res_list_l.index('71'), max_res_list_l.index('75'))

    maps = []
    coord_h = [cdr1_coord_h, beta11_coord_h, beta12_coord_h, cdr2_coord_h, beta13_coord_h, beta21_coord_h, beta22_coord_h, alpha_coord_h, beta14_coord_h, cdr3_coord_h]
    coord_l = [cdr1_coord_l, beta11_coord_l, beta12_coord_l, cdr2_coord_l, beta21_coord_l, beta22_coord_l, beta13_coord_l, cdr3_coord_l]
    coord = coord_h + coord_l

    for i in range(len(coord)):
        maps.append(epsilon[:, coord[i]])

    return np.array(coord, dtype=object), maps