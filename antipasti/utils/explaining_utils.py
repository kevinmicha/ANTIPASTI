import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import umap

from copy import deepcopy
from matplotlib.colors import CenteredNorm
import matplotlib.patches as patches
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

from antipasti.utils.biology_utils import remove_nanobodies

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

def get_epsilon(preprocessed_data, model, mode='general'):
    r"""Returns a map ``epsilon`` (ϵ) such that the predicted affinity of x + ϵ is always greater than that of x.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    mode: str
        Choose between ``general`` and ``extreme``.

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
        new_h = np.multiply(-np.clip(each_img_enl[j], a_min=-np.inf, a_max=0), np.sign(train_x[j]))
        new_l = np.multiply(np.clip(each_img_enl[j], a_min=0, a_max=np.inf), np.sign(train_x[j]))
        if j == 0 or mode == 'general':
            current_h = new_h
            current_l = new_l
        else:
            current_h = np.where(np.abs(current_h)>np.abs(new_h), current_h, new_h)
            current_l = np.where(np.abs(current_l)>np.abs(new_l), current_l, new_l)
        high_aff.append(current_h)
        low_aff.append(current_l)

    if mode == 'general':
        epsilon = deepcopy(np.mean(high_aff, axis=0) - np.mean(low_aff, axis=0))
    else:
        epsilon = high_aff[-1] - low_aff[-1]

    return epsilon

def get_test_contribution(preprocessed_data, model):
    r"""Returns a map that reveals how to mutate a given test antibody in order to increase its affinity.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.

    """
    test_x = preprocessed_data.test_x
    input_shape = preprocessed_data.test_x.shape[-1]
    n_filters = model.n_filters
    each_img_enl = np.zeros((input_shape, input_shape))
    size_le = int(np.sqrt(model.fc1.weight.data.numpy().shape[-1] / n_filters))

    inter_filter_item = model(torch.from_numpy(test_x.reshape(1, 1, input_shape, input_shape).astype(np.float32)))[1].detach().numpy()
    for i in range(n_filters):
        each_img_enl += cv2.resize(np.multiply(inter_filter_item[0,i], model.fc1.weight.data.numpy().reshape(n_filters, size_le**2)[i].reshape(size_le, size_le)), dsize=(input_shape, input_shape))
    contribution = np.multiply(-np.clip(each_img_enl, a_min=-np.inf, a_max=0), np.sign(test_x)) - np.multiply(np.clip(each_img_enl, a_min=0, a_max=np.inf), np.sign(test_x))
    
    return contribution

def plot_map_with_regions(preprocessed_data, map, title='Normal mode correlation map', interactive=False):
    r"""Maps the residues to the antibody regions and plots the normal mode correlation map.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    map: numpy.ndarray
        A normal mode correlation map.
    title: str
        The image title.
    interactive: bool
        Set to ``True`` when running a script or Pytest.

    """
    # Font sizes
    title_size = 48
    font_size = 32

    # Defining the region boundaries
    mrlh = preprocessed_data.max_res_list_h
    mrll = preprocessed_data.max_res_list_l

    subgroup_boundaries_h = [mrlh.index('3'), mrlh.index('26'), mrlh.index('33'), mrlh.index('39'), mrlh.index('45'), mrlh.index('51'),
                        mrlh.index('52'), mrlh.index('57'), mrlh.index('61'), mrlh.index('67'), mrlh.index('72'),
                        mrlh.index('75'), mrlh.index('82'), mrlh.index('84'), mrlh.index('87'), mrlh.index('89'),
                        mrlh.index('95'), mrlh.index('103'), mrlh.index('112')+1]
    subgroup_boundaries_l = [mrll.index('3'), mrll.index('24'), mrll.index('35'), mrll.index('38'), mrll.index('44'), mrll.index('48'),
                        mrll.index('50'), mrll.index('57'), mrll.index('62'), mrll.index('65'), mrll.index('71'),
                        mrll.index('75'), mrll.index('85'), mrll.index('89'), mrll.index('98'), mrll.index('106')+1]
    labels_h = ['F-START', 'CDR1', '\u03B211', '', '\u03B212', '', 'CDR2', '\u03B213', '', '\u03B221', '', '\u03B222',
            '', '\u03B1', '', '\u03B214', 'CDR3', 'F-END', '']
    labels_l = ['F-START', 'CDR1', '\u03B211', '', '\u03B212', '', 'CDR2', '', '\u03B221', '', '\u03B222',
            '', '\u03B213', 'CDR3', 'F-END']
    subgroup_boundaries = subgroup_boundaries_h + [x+mrlh.index('112')+1 for x in subgroup_boundaries_l]
    labels = labels_h + labels_l
    fig = plt.figure(figsize=(20, 20))

    # Plotting the normal mode correlation map
    plt.imshow(map, origin='lower', cmap='seismic', norm=CenteredNorm())
    cbar = plt.colorbar(fraction=0.045)

    # Set the font size of the colorbar
    cbar.ax.tick_params(labelsize=18)

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
    plt.text(-14, subgroup_boundaries_h[-1]/2, 'Heavy chain', ha='center', va='center', color='black', rotation='vertical', size=16)
    plt.text(-14, subgroup_boundaries_h[-1]+subgroup_boundaries_l[-1]/2, 'Light chain', ha='center', va='center', color='black', rotation='vertical', size=16)
    # Adding labels for the chains on the x-axis
    plt.text(subgroup_boundaries_h[-1]/2, -8.75, 'Heavy chain', ha='center', va='center', color='black', size=16)
    plt.text(subgroup_boundaries_h[-1]+subgroup_boundaries_l[-1]/2, -8.75, 'Light chain', ha='center', va='center', color='black', size=16)

    # Adjusting the axis limits and labels
    plt.xlim(-16.5, 278.5)
    plt.ylim(-10.5, 278.5)
    plt.xticks([])
    plt.yticks([])

    # Adding labels and title
    plt.xlabel('Residues', size=font_size)
    plt.ylabel('Residues', size=font_size)
    plt.title(title, size=title_size)

    plt.show(block=False)
    if interactive:
        plt.pause(3)
        plt.close('all')

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
    labels: list
        The antibody regions in order.

    """
    mrlh = preprocessed_data.max_res_list_h
    mrll = preprocessed_data.max_res_list_l

    maps = []
    subgroup_boundaries_h = [mrlh.index('3'), mrlh.index('26'), mrlh.index('33'), mrlh.index('39'), mrlh.index('45'), mrlh.index('51'),
                        mrlh.index('52'), mrlh.index('57'), mrlh.index('61'), mrlh.index('67'), mrlh.index('72'),
                        mrlh.index('75'), mrlh.index('82'), mrlh.index('84'), mrlh.index('87'), mrlh.index('89'),
                        mrlh.index('95'), mrlh.index('103'), mrlh.index('112')+1]
    subgroup_boundaries_l = [mrll.index('3'), mrll.index('24'), mrll.index('35'), mrll.index('38'), mrll.index('44'), mrll.index('48'),
                        mrll.index('50'), mrll.index('57'), mrll.index('62'), mrll.index('65'), mrll.index('71'),
                        mrll.index('75'), mrll.index('85'), mrll.index('89'), mrll.index('98'), mrll.index('106')+1]
    labels_h = ['F-START', 'CDR-H1', '\u03B211', '', '\u03B212', '', 'CDR-H2', '\u03B213', '', '\u03B221', '', '\u03B222',
            '', '\u03B1', '', '\u03B214', 'CDR-H3', 'F-END']
    labels_l = ['F-START', 'CDR-L1', '\u03B211', '', '\u03B212', '', 'CDR-L2', '', '\u03B221', '', '\u03B222',
            '', '\u03B213', 'CDR-L3', 'F-END']
    
    coord_h = [range(subgroup_boundaries_h[i], subgroup_boundaries_h[i+1]) for i in range(len(subgroup_boundaries_h)-1)]
    coord_l = [range(subgroup_boundaries_l[i], subgroup_boundaries_l[i+1]) for i in range(len(subgroup_boundaries_l)-1)]
    coord = coord_h + [range(cl[0]+mrlh.index('112')+1, cl[-1]+mrlh.index('112')+1) for cl in coord_l]
    labels = labels_h + labels_l

    for i in range(len(coord)):
        maps.append(epsilon[:, coord[i]])

    return np.array(coord, dtype=object), maps, labels

def compute_umap(preprocessed_data, model, scheme='heavy_species', regions='paired_hl', categorical=True, include_ellipses=False, numerical_values=None, external_cdict=None, interactive=False, exclude_nanobodies=False):
    r"""Performs UMAP dimensionality reduction calculations.

    Parameters
    ----------
    preprocessed_data: antipasti.model.model.Preprocessing
        The ``Preprocessing`` class.
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    scheme: str
        Category of the labels or values appearing in the UMAP representation.
    regions: str
        Choose between ``paired_hl`` (heavy chain, light chain and their interactions) and ``heavy`` (heavy chain only).
    categorical: bool
        ``True`` if ``scheme`` is categorical.
    include_ellipses: bool
        ``True`` if ellipses comprising three quarters of the points of a given class are included.
    numerical_values: list
        A list of values or entries should be provided if data external to SAbDab is used.
    external_cdict: dictionary
        Option to provide an external dictionary of the UMAP labels.
    interactive: bool
        Set to ``True`` when running a script or Pytest.
    remove_nanobodies: bool
        Set to ``True`` to exclude nanobodies from the UMAP plot.

    """
    train_x = preprocessed_data.train_x
    input_shape = preprocessed_data.test_x.shape[-1]

    if regions == 'paired_hl':
        reducer = umap.UMAP(random_state=32, min_dist=0.1, n_neighbors=90) # Paired-HL
        umap_shape = input_shape
    else:
        reducer = umap.UMAP(random_state=32, min_dist=0.15, n_neighbors=16) # Heavy
        umap_shape = len(preprocessed_data.max_res_list_h)

    labels = []
    colours = []
    each_img_enl = np.zeros((train_x.shape[0], umap_shape**2))
    n_filters = model.n_filters
    size_le = int(np.sqrt(model.fc1.weight.data.numpy().shape[-1] / n_filters))
    pdb_codes = preprocessed_data.labels
    db = pd.read_csv(preprocessed_data.data_path+'sabdab_summary_all.tsv', sep='\t')
    if scheme in db.columns:
        db = db.loc[:,['pdb', scheme]]

    # Obtaining the labels and the output layer representations
    for j in range(train_x.shape[0]):
        if scheme in db.columns:
            labels.append(str(db[db['pdb'] == pdb_codes[j]].iloc[-1][scheme]))
        inter_filter_item = model(torch.from_numpy(train_x[j].reshape(1, 1, input_shape, input_shape).astype(np.float32)))[1].detach().numpy()
        for i in range(n_filters):
            each_img_enl[j] += cv2.resize(np.multiply(inter_filter_item[0,i], model.fc1.weight.data.numpy().reshape(n_filters, size_le**2)[i].reshape(size_le, size_le)), dsize=(input_shape, input_shape))[:umap_shape, :umap_shape].reshape((umap_shape**2))
    
    # UMAP fitting 
    scaled_each_img = StandardScaler().fit_transform(each_img_enl)
    embedding = reducer.fit_transform(scaled_each_img)

    if exclude_nanobodies:
        pdb_codes, _, embedding, labels, numerical_values = remove_nanobodies(pdb_codes, train_x, embedding, labels, numerical_values)

    if categorical:
        if scheme == 'light_subclass':
            cdict = {'IGKV1': 0,
                'IGKV2': 1,
                'IGKV3': 2,
                'IGKV4': 3,
                'IGKV5': 4,
                'IGKV6': 5,
                'IGKV7': 6,
                'IGKV8': 7,
                'IGKV9': 8,
                'IGKV10': 9,
                'IGKV14': 10,
                'IGLV1': 11,
                'IGLV2': 12,
                'IGLV6': 13,
                'Other': 14,}
            scheme = 'Light chain V gene family'

        elif scheme == 'heavy_subclass':
            cdict = {'IGHV1': 0,
                'IGHV2': 1,
                'IGHV3': 2,
                'IGHV4': 3,
                'IGHV5': 4,
                'IGHV6': 5,
                'IGHV7': 6,
                'Other': 7,}
            scheme = 'Heavy chain V gene family'

        elif scheme == 'heavy_species' or scheme == 'light_species':
            cdict = {'homo sapiens': 0,
                'mus musculus': 1,
                'Other': 2}
            scheme = 'Antibody species'

        elif scheme == 'light_ctype':
            cdict = {'Kappa': 0,
                'Lambda': 1,
                'unknown': 2,
                'NA': 3,
                'Other': 4,}
            scheme = 'Type of light chain'

        elif scheme == 'antigen_type':
            cdict = {'protein': 0,
                'peptide': 1,
                'Hapten': 2,
                'protein | protein': 3,
                'carbohydrate': 4,
                'Other': 5}
            scheme = 'Type of antigen'
        else:
            cdict = external_cdict

        for i in range(len(labels)):
            if labels[i] in cdict:
                colours.append(cdict[labels[i]])
            else:
                colours.append(cdict['Other'])
                labels[i] = 'Other'
    else:
        cdict = None
        deleted_items = 0
        for i, item in enumerate(numerical_values):
            if isinstance(item, (int, float, np.int64, np.float32)):
                colours.append(item)
            elif item.replace('.', '').isnumeric():
                colours.append(float(item))
            else:
                embedding = np.delete(embedding, i-deleted_items, axis=0)
                pdb_codes = np.delete(pdb_codes, i-deleted_items, axis=0)
                deleted_items += 1
    plot_umap(embedding=embedding, colours=colours, scheme=scheme, pdb_codes=pdb_codes, categorical=categorical, include_ellipses=include_ellipses, cdict=cdict, interactive=interactive)

def plot_umap(embedding, colours, scheme, pdb_codes, categorical=True, include_ellipses=False, cdict=None, interactive=False):
    r"""Plots UMAP maps.

    Parameters
    ----------
    embedding: numpy.ndarray
        The output layer representations after dimensionality reduction.
    colours: list
        The data points labels or values.
    scheme: str
        Category of the labels or values appearing in the UMAP representation.
    pdb_codes: list
        The PDB codes of the antibodies.
    categorical: bool
        ``True`` if ``scheme`` is categorical.
    include_ellipses: bool
        ``True`` to include ellipses comprising 85% of the points of a given class.
    interactive: bool
        Set to ``True`` when running a script or Pytest.

    """
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot()

    if categorical:
        cmap = matplotlib.colormaps.get_cmap('tab10')
    else:
        cmap = matplotlib.colormaps.get_cmap('Purples')
    unique_colours = list(set(colours))
    norm = plt.Normalize(np.min(colours), np.max(colours))
    legend_patches = [patches.Patch(color=cmap(norm(color))) for color in unique_colours]
    im = ax.scatter(embedding[:, 0], embedding[:, 1] , s=50, c=colours, cmap=cmap)

    for i in range(len(pdb_codes)):
        if i % 10 == 0:
            ax.annotate(pdb_codes[i], (embedding[i, 0], embedding[i, 1]), size=8)

    if include_ellipses:
        # Inverse of the chi-squared CDF
        conf_level = 0.85
        inv_chi2 = chi2.ppf(conf_level, df=2)
        ellipses = []  # Store ellipse information

        for label in unique_colours:
            label_points = embedding[np.array(colours) == label]  # Subset of UMAP points for a specific label
            n_points = len(label_points)
            
            # Calculate the centroid using all the points of a class
            center = np.mean(label_points, axis=0)
            covariance = np.cov(label_points.T)
            # Calculate the distance of each point
            dist = np.sum(np.square(label_points - center), axis=1)
            
            # Sort the points based on the distance
            sorted_indices = np.argsort(dist)
            
            # Calculate the number of points to include within the ellipse
            n_inside = int(np.ceil(n_points * conf_level))
            
            # Select the points that fall within the ellipse
            inside_points = label_points[sorted_indices[:n_inside]]
            
            # Recalculate the mean and covariance using only the inside points
            center = np.mean(inside_points, axis=0)
            covariance = np.cov(inside_points.T)
            
            # Calculate the eigenvalues and eigenvectors of the covariance matrix again
            eigenvalues, eigenvectors = np.linalg.eig(covariance)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            
            # Calculate the scaling factor for the ellipse based on the eigenvalues again
            scale_factor = np.sqrt(inv_chi2)
            
            # Calculate the radius of the ellipse based on the eigenvalues and the scaling factor
            radius = np.sqrt(eigenvalues) * scale_factor
            
            ellipse = patches.Ellipse(xy=center, width=2 * radius[0], height=2 * radius[1],
                                    angle=angle, fill=False, linewidth=3, alpha=0.7, color=cmap(norm(label)))
            ellipses.append(ellipse)

        for i, ellipse in enumerate(ellipses):
            if list(cdict.keys())[i] not in ['unknown', 'Other']:
                ax.add_patch(ellipse)

    if categorical:
        legend1 = ax.legend(legend_patches, cdict.keys(), loc='best')
    else:
        legend1 = ax.legend(legend_patches[:10], set(colours), loc='best')
    ax.add_artist(legend1)

    ax.set_title(scheme, size=18)
    ax.set_xlabel('UMAP 1', size=16)
    ax.set_ylabel('UMAP 2', size=16)
    plt.show(block=False)
    if interactive:
        plt.pause(3)
        plt.close('all')