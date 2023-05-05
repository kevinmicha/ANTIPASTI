import cv2
import numpy as np
import torch

from copy import deepcopy

def get_maps_of_interest(preprocessed_data, learnt_filter, affinity_thr=-8):
    r"""Post-processes both raw data and results to obtain maps of interest.

    Parameters
    ----------
    preprocessed_data: nmacnn.model.model.Preprocessing
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
    preprocessed_data: nmacnn.model.model.Preprocessing
        The ``Preprocessing`` class.
    model: nmacnn.model.model.NormalModeAnalysisCNN
        The model class, i.e., ``NormalModeAnalysisCNN``.
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
        high_aff.append(np.multiply(-np.clip(each_img_enl[j], a_min=-np.inf, a_max=0), cv2.resize(train_x[j], dsize=(input_shape, input_shape))))
        low_aff.append(np.multiply(np.clip(each_img_enl[j], a_min=0, a_max=np.inf), cv2.resize(train_x[j], dsize=(input_shape, input_shape))))

    true_filter = deepcopy(np.mean(high_aff, axis=0) - np.mean(low_aff, axis=0))
    epsilon = np.multiply(np.abs(np.sign(mean_diff_image[0])), true_filter)

    return epsilon

def map_residues_to_regions(preprocessed_data, epsilon):
    r"""Maps the residues to the antibody regions.

    Parameters
    ----------
    preprocessed_data: nmacnn.model.model.Preprocessing
        The ``Preprocessing`` class.
    epsilon: numpy.ndarray
        A map ``epsilon`` (ϵ) such that the predicted affinity of x + ϵ is always greater than that of x.

    Returns
    -------
    coord: numpy.ndarray
        The coordinates of the antibody regions.
    maps: list
        The maps of the antibody regions coming from ``epsilon``.
    ticks: list
        Locations where to place the ticks on the antibody regions plots.
    ticks_labels: list
        Labels of the antibody region limits.
    titles: list
        Antibody region labels.
    rmax = float
        The largest value of ``epsilon``.

    """
    max_res_list_h = preprocessed_data.max_res_list_h
    max_res_list_l = preprocessed_data.max_res_list_l
    rmax = np.abs(epsilon).max()

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

    ticks = [cdr1_coord_h[0], (cdr1_coord_h[-1]+beta11_coord_h[0])/2, beta11_coord_h[-1], beta12_coord_h[0], beta12_coord_h[-1], cdr2_coord_h[0], (cdr2_coord_h[-1]+beta13_coord_h[0])/2, beta13_coord_h[-1], beta21_coord_h[0], beta21_coord_h[-1],
            beta22_coord_h[0], beta22_coord_h[-1], beta14_coord_h[0], (beta14_coord_h[-1]+cdr3_coord_h[0])/2, cdr3_coord_h[-1]]
    maps = []
    ticks_labels = ['CDR-H1_i', 'CDR-H1_f & Beta11_i', 'Beta11_f', 'Beta12_i', 'Beta12_f', 'CDR-H2_i', 'CDR-H2_f & Beta13_i', 'Beta13_f', 'Beta21_i', 'Beta21_f', 'Beta22_i', 'Beta22_f', 'Beta14_i', 'Beta14_f & CDR-H3_i', 'CDR-H3_f']

    coord = [cdr1_coord_h, beta11_coord_h, beta12_coord_h, cdr2_coord_h, beta13_coord_h, beta21_coord_h, beta22_coord_h, alpha_coord_h, beta14_coord_h, cdr3_coord_h]

    for i in range(len(coord)):
        maps.append(epsilon[:, coord[i]])

    titles = deepcopy(ticks_labels)

    del titles[1::2]
    for i in titles:
        idx = titles.index(i)
        if '&' in titles[idx]:
            titles[idx] = i.split('&')[0].strip()
            titles.insert(idx+1, i.split('&')[1].strip())
        titles[idx] = titles[idx][:-2]
    titles.insert(-2, 'Alpha helix')

    return np.array(coord, dtype=object), maps, ticks, ticks_labels, titles, rmax

def compute_change_in_kd(preprocessed_data, model, weights, coord, maps):
    r"""Prints the percentage change in Kd when adding epsilon.

    Parameters
    ----------
    preprocessed_data: nmacnn.model.model.Preprocessing
        The ``Preprocessing`` class.
    model: nmacnn.model.model.NormalModeAnalysisCNN
        The model class, i.e., ``NormalModeAnalysisCNN``.
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