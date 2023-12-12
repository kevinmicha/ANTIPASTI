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

from antipasti.utils.biology_utils import remove_nanobodies, extract_mean_region_lengths

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
    mean_learnt = cv2.resize(-learnt_filter, dsize=(input_shape, input_shape))
    mean_image = np.mean(train_x, axis=0).reshape(input_shape, input_shape)
    mean_diff_image = np.mean(high_aff, axis=0) - np.mean(low_aff, axis=0)

    return mean_learnt, mean_image, mean_diff_image

def get_output_representations(preprocessed_data, model):
    r"""Returns maps that reveal the important residue interactions for the binding affinity. We call them 'output layer representations'.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.

    """
    input_shape = preprocessed_data.test_x.shape[-1]
    each_img_enl = np.zeros((preprocessed_data.train_x.shape[0], input_shape**2))
    size_le = int(np.sqrt(model.fc1.weight.data.numpy().shape[-1] / model.n_filters))
    offset = np.zeros((input_shape**2))

    inter_filter_off = model(torch.from_numpy(np.zeros((input_shape, input_shape)).reshape(1, 1, input_shape, input_shape).astype(np.float32)))[1].detach().numpy()
    for i in range(model.n_filters):
        offset += cv2.resize(np.multiply(inter_filter_off[0,i], model.fc1.weight.data.numpy().reshape(model.n_filters, size_le**2)[i].reshape(size_le, size_le)), dsize=(input_shape, input_shape)).reshape((input_shape**2))

    for j in range(preprocessed_data.train_x.shape[0]):
        inter_filter_item = model(torch.from_numpy(preprocessed_data.train_x[j].reshape(1, 1, input_shape, input_shape).astype(np.float32)))[1].detach().numpy()
        for i in range(model.n_filters):
            each_img_enl[j] += (size_le**2/input_shape**2) * cv2.resize(np.multiply(inter_filter_item[0,i], model.fc1.weight.data.numpy().reshape(model.n_filters, size_le**2)[i].reshape(size_le, size_le)), dsize=(input_shape, input_shape)).reshape((input_shape**2))
        each_img_enl[j] -= offset

    return each_img_enl

def get_test_contribution(preprocessed_data, model):
    r"""Returns a map that reveals the important residue interactions for the binding affinity.

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
        each_img_enl -= cv2.resize(np.multiply(inter_filter_item[0,i], model.fc1.weight.data.numpy().reshape(n_filters, size_le**2)[i].reshape(size_le, size_le)), dsize=(input_shape, input_shape))
    
    return each_img_enl

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
        Set to ``True`` when running a script or ``pytest``.

    """
    # Font sizes
    title_size = 42
    font_size = 32

    # Defining the region boundaries
    mrlh = preprocessed_data.max_res_list_h
    mrll = preprocessed_data.max_res_list_l

    subgroup_boundaries_h = [mrlh.index('1'), mrlh.index('26'), mrlh.index('33'), mrlh.index('52'), mrlh.index('57'), mrlh.index('95'), mrlh.index('103'), mrlh.index('113')+1]
    subgroup_boundaries_l = [mrll.index('1'), mrll.index('24'), mrll.index('35'), mrll.index('50'), mrll.index('57'), mrll.index('89'), mrll.index('98'), mrll.index('107')+1]
    labels_h = ['FR-H1', 'CDR-H1', 'FR-H2', 'CDR-H2', 'FR-H3', 'CDR-H3', 'FR-H4']
    labels_l = ['FR-L1', 'CDR-L1', 'FR-L2', 'CDR-L2', 'FR-L3', 'CDR-L3', 'FR-L4']
    subgroup_boundaries = subgroup_boundaries_h[:-1] + [x+mrlh.index('113')+1 for x in subgroup_boundaries_l]
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
            c = 'deeppink'
        else:
            c = 'orange'
            
        # Adding rectangles for the regions
        rect = plt.Rectangle((start_index - 0.5, -6.5), end_index - start_index, 6, edgecolor='black', facecolor=c, alpha=0.7)
        plt.gca().add_patch(rect)

        rect = plt.Rectangle((-12.5, start_index - 0.5), 12, end_index - start_index, edgecolor='black', facecolor=c, alpha=0.7)
        plt.gca().add_patch(rect)

        # Add labels for the regions on the y-axis
        plt.text(-6, label_position-0.25, labels[i], ha='center', va='center', color='black', size=10)
        # Add labels for the regions on the x-axis
        plt.text(label_position, -3.7, labels[i], ha='center', va='center', color='black', size=9)

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
    plt.text(subgroup_boundaries_h[-1]/2, -9, 'Heavy chain', ha='center', va='center', color='black', size=12)
    plt.text(subgroup_boundaries_h[-1]+subgroup_boundaries_l[-1]/2, -9, 'Light chain', ha='center', va='center', color='black', size=12)

    # Adjusting the axis limits and labels
    plt.xlim(-16.5, 290.5)
    plt.ylim(-10.5, 290.5)
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

def compute_umap(preprocessed_data, model, scheme='heavy_species', categorical=True, include_ellipses=False, numerical_values=None, external_cdict=None, interactive=False, exclude_nanobodies=False):
    r"""Performs UMAP dimensionality reduction calculations.

    Parameters
    ----------
    preprocessed_data: antipasti.model.model.Preprocessing
        The ``Preprocessing`` class.
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    scheme: str
        Category of the labels or values appearing in the UMAP representation.
    categorical: bool
        ``True`` if ``scheme`` is categorical.
    include_ellipses: bool
        ``True`` if ellipses comprising three quarters of the points of a given class are included.
    numerical_values: list
        A list of values or entries should be provided if data external to SAbDab is used.
    external_cdict: dictionary
        Option to provide an external dictionary of the UMAP labels.
    interactive: bool
        Set to ``True`` when running a script or ``pytest``.
    exclude_nanobodies: bool
        Set to ``True`` to exclude nanobodies from the UMAP plot.

    """
    train_x = preprocessed_data.train_x
    input_shape = preprocessed_data.test_x.shape[-1]

    reducer = umap.UMAP(random_state=32, min_dist=0.1, n_neighbors=90) # Paired-HL

    labels = []
    colours = []
    pdb_codes = preprocessed_data.labels
    db = pd.read_csv(preprocessed_data.data_path+'sabdab_summary_all.tsv', sep='\t')
    if scheme in db.columns:
        db = db.loc[:,['pdb', scheme]]

    # Obtaining the labels and the output layer representations
    for j in range(train_x.shape[0]):
        if scheme in db.columns:
            labels.append(str(db[db['pdb'] == pdb_codes[j]].iloc[-1][scheme]))
    each_img_enl = get_output_representations(preprocessed_data, model)

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

    return colours, pdb_codes


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
    cdict: dictionary
        External dictionary of the UMAP labels.
    interactive: bool
        Set to ``True`` when running a script or ``pytest``.

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
        if i % 1 == 0:
            ax.annotate(pdb_codes[i], (embedding[i, 0], embedding[i, 1]), size=8)

    if include_ellipses:
        # Inverse of the chi-squared CDF
        conf_level = 0.85
        inv_chi2 = chi2.ppf(conf_level, df=2)
        ellipses = []  # Store ellipse information

        for label in unique_colours:
            label_points = embedding[np.array(colours) == label]  # Subset of UMAP points for a specific label
            n_points = len(label_points)
            
            # Centroid and then sort
            center = np.mean(label_points, axis=0)
            covariance = np.cov(label_points.T)
            dist = np.sum(np.square(label_points - center), axis=1)
            sorted_indices = np.argsort(dist)
            
            # Calculate the number of points to include within the ellipse
            n_inside = int(np.ceil(n_points * conf_level))
            inside_points = label_points[sorted_indices[:n_inside]]
            
            # Recalculate the mean and covariance using only the inside points
            center = np.mean(inside_points, axis=0)
            covariance = np.cov(inside_points.T)
            
            # Calculate the eigenvalues and eigenvectors of the cov matrix (again)
            eigenvalues, eigenvectors = np.linalg.eig(covariance)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            
            scale_factor = np.sqrt(inv_chi2)
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

def plot_region_importance(importance_factor, importance_factor_ob, antigen_type, mode='region', interactive=False):
    r"""Plots ranking of important regions.

    Parameters
    ----------
    importance_factor: list
        Measure of importance (0-100) for each antibody region.
    importance_factor_ob: list
        Measure of importance (0-100) for each antibody region attributable to off-block correlations. This can be inter-region or inter-chain depending on the selected ``mode``.
    antigen_type: int
        Plot corresponding to antigens of a given type. These can be proteins (0), haptens (1), peptides (2) or carbohydrates (3).
    mode: str
        ``region`` to explicitely show which correlations are inter/intra-region (likewise for ``chain``).
    interactive: bool
        Set to ``True`` when running a script or ``pytest``.

    """

    labels = ['FR-H1', 'CDR-H1', 'FR-H2', 'CDR-H2', 'FR-H3', 'CDR-H3', 'FR-H4', 'FR-L1', 'CDR-L1', 'FR-L2', 'CDR-L2', 'FR-L3', 'CDR-L3', 'FR-L4']
    mapping_dict = {0: 0, 1: 2, 2: 1, 3: 5}

    sorted_indices = np.argsort(importance_factor)[::-1]  # Reverse order
    cmap = matplotlib.colormaps.get_cmap('tab10')

    # Create bars for each class
    fig, ax = plt.subplots()
    y_pos = np.arange(len(labels))

    bar1 = ax.barh(y_pos, np.array(importance_factor_ob)[sorted_indices], align='center', color=cmap(mapping_dict[antigen_type]), label=f'Inter-{mode}')
    bar2 = ax.barh(y_pos, np.array([importance_factor[i]-importance_factor_ob[i] for i in range(len(labels))])[sorted_indices], align='center', alpha=0.6, left=np.array(importance_factor_ob)[sorted_indices], color=cmap(mapping_dict[antigen_type]), label=f'Intra-{mode}')
    
    ax.set_xlabel('Importance (%)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[np.argsort(importance_factor)[::-1][i]] for i in range(len(labels))])
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.tick_params(axis='x', which='major', bottom=True, right=True, size=3.5)

    for i, label in enumerate(ax.get_yticklabels()):
        if np.argsort(importance_factor)[::-1][i] < 7:
            color = 'green'
        else:
            color = '#333333'
        label.set_color(color)

    plt.tight_layout()
    ax.legend()
    plt.show()
    if interactive:
        plt.pause(3)
        plt.close('all')

def add_region_based_on_range(list_residues):
    r"""Given a list of residues in Chothia numbering, this function adds the corresponding regions in brackets for each of them."""
    output_list_residues = []

    new_coord = np.array([range(0, 26), range(26, 38), range(38, 57), range(57, 67), range(67, 116), range(116, 142),
                         range(142, 153), range(153, 176), range(176, 195), range(195, 210), range(210, 225),
                         range(225, 265), range(265, 279), range(279, 292)], dtype=object)

    regions = ['FR-H1', 'CDR-H1', 'FR-H2', 'CDR-H2', 'FR-H3', 'CDR-H3', 'FR-H4', 'FR-L1', 'CDR-L1', 'FR-L2', 'CDR-L2', 'FR-L3', 'CDR-L3', 'FR-L4']

    for index, element in enumerate(list_residues):
        for i, r in enumerate(new_coord):
            if index in r:
               output_list_residues.append(element+' (' + regions[i] + ')')

    return output_list_residues

def plot_residue_importance(preprocessed_data, importance_factor, antigen_type, interactive=False):
    r"""Plots ranking of important residues.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    importance_factor: list
        Measure of importance (0-100) for each antibody residue.
    antigen_type: int
        Plot corresponding to antigens of a given type. These can be proteins (0), haptens (1), peptides (2) or carbohydrates (3).
    interactive: bool
        Set to ``True`` when running a script or ``pytest``.

    """

    res_labels = add_region_based_on_range(preprocessed_data.max_res_list_h+preprocessed_data.max_res_list_l)
    mapping_dict = {0: 0, 1: 2, 2: 1, 3: 5}
    cmap = matplotlib.colormaps.get_cmap('tab10')

    fig, ax = plt.subplots()
    y_pos = np.arange(len(res_labels[:30]))
    bar1 = ax.barh(y_pos, sorted(importance_factor, reverse=True)[:30], align='center', alpha=0.9, color=cmap(mapping_dict[antigen_type]))

    # Show top 30
    ax.set_xlabel('Importance (%)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([res_labels[np.argsort(importance_factor)[::-1][i]][:30] for i in range(len(res_labels[:30]))], fontsize=9.5)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.tick_params(axis='x', which='major', bottom=True, right=True, size=3.5)

    for i, label in enumerate(ax.get_yticklabels()):
        if np.argsort(importance_factor)[::-1][i] < len(preprocessed_data.max_res_list_h): 
            color = 'green'
        else:
            color = '#333333'
        label.set_color(color)

    plt.tight_layout()
    plt.show()

    if interactive:
        plt.pause(3)
        plt.close('all')


def get_colours_ag_type(preprocessed_data):
    r"""Returns a different colour according to the antigen type.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    
    """

    cluster_according_to = 'antigen_type'
    db = pd.read_csv(preprocessed_data.data_path+'sabdab_summary_all.tsv', sep='\t')

    clusters = []
    for i in range(len(preprocessed_data.labels)):
        clusters.append(str(db[db['pdb'] == preprocessed_data.labels[i]].iloc[0][cluster_according_to]))

    cdict = {'protein': 0,
            'Hapten': 1,
            'peptide': 2,
            'carbohydrate': 3,
            'nucleic-acid': 4,
            'protein | protein': 5,
            'Other': 6}

    colours = []
    for i in range(len(clusters)):
        if clusters[i] in cdict:
            colours.append(cdict[clusters[i]])
        else:
            colours.append(cdict['Other'])
            clusters[i] = 'Other'

    return colours 

def compute_region_importance(preprocessed_data, model, type_of_antigen, nanobodies, mode='region', interactive=False):
    r"""Computes the importance factors (0-100) of all the Fv antibody regions.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    type_of_antigen: int
        Choose between: proteins (0), haptens (1), peptides (2) or carbohydrates (3).
    nanobodies: list
        PDB codes of nanobodies in the dataset.
    mode: str
        ``region`` to explicitely calculate which correlations are inter/intra-region (likewise for ``chain``).
    interactive: bool
        Set to ``True`` when running a script or ``pytest``.

    """

    colours = get_colours_ag_type(preprocessed_data)
    each_img_enl = get_output_representations(preprocessed_data, model)

    train_x = preprocessed_data.train_x
    input_shape = preprocessed_data.test_x.shape[-1]

    colours = [0 if c == 5 else c for c in colours]
    all_mse_without_region = []
    all_mse_without_region_intra = []
    all_mse_without_region_ob = []

    new_coord = np.array([range(0, 26), range(26, 38), range(38, 57), range(57, 67), range(67, 116), range(116, 142), range(142, 153),
                        range(153, 176), range(176, 195), range(195, 210), range(210, 225), range(225, 265), range(265, 279), range(279, 292)], dtype=object)
    
    for j in range(len(new_coord)+1):
        train_y_ = np.array([preprocessed_data.train_y[i] for i in range(train_x.shape[0]) if colours[i] == type_of_antigen and preprocessed_data.labels[i] not in nanobodies])
        if j != len(new_coord):
            
            sums_without_region = np.array([
                each_img_enl[i].reshape((input_shape, input_shape)).sum()-(each_img_enl[i].reshape((input_shape, input_shape))[new_coord[j][0]:new_coord[j][-1] + 1, :]).sum()
                for i in range(train_x.shape[0]) if colours[i] == type_of_antigen and preprocessed_data.labels[i] not in nanobodies])

            if mode == 'region':
                sums_without_region_divided = np.array([
                        each_img_enl[i].reshape((input_shape, input_shape)).sum()-np.array([(each_img_enl[i].reshape((input_shape, input_shape))[new_coord[j][0]:new_coord[j][-1] + 1, new_coord[0][0]:new_coord[j][0]]).sum()
                                                                    +(each_img_enl[i].reshape((input_shape, input_shape))[new_coord[j][0]:new_coord[j][-1] + 1, new_coord[j][-1] + 1:new_coord[-1][-1] + 1]).sum(),
                                                                    (each_img_enl[i].reshape((input_shape, input_shape))[new_coord[j][0]:new_coord[j][-1] + 1, new_coord[j][0]:new_coord[j][-1] + 1]).sum()])
                        for i in range(train_x.shape[0]) if colours[i] == type_of_antigen and preprocessed_data.labels[i] not in nanobodies])

                all_mse_without_region_intra.append(np.mean((sums_without_region_divided[:,1] - train_y_)**2))
                all_mse_without_region_ob.append(np.mean((sums_without_region_divided[:,0] - train_y_)**2))
            
            else:
                sums_without_region_divided = np.array([
                        each_img_enl[i].reshape((input_shape, input_shape)).sum()-np.array([(each_img_enl[i].reshape((input_shape, input_shape))[new_coord[j][0]:new_coord[j][-1] + 1, :len(preprocessed_data.max_res_list_h)]).sum(),
                                                                    (each_img_enl[i].reshape((input_shape, input_shape))[new_coord[j][0]:new_coord[j][-1] + 1, len(preprocessed_data.max_res_list_h):]).sum()])
                        for i in range(train_x.shape[0]) if colours[i] == type_of_antigen and preprocessed_data.labels[i] not in nanobodies])

                index = 0 if j < 7 else 1
                all_mse_without_region_intra.append(np.mean((sums_without_region_divided[:, index] - train_y_)**2))
                all_mse_without_region_ob.append(np.mean((sums_without_region_divided[:, 1 - index] - train_y_)**2))
        else:
            sums_without_region = np.array([
                    each_img_enl[i].reshape((input_shape, input_shape)).sum()
                    for i in range(train_x.shape[0]) if colours[i] == type_of_antigen and preprocessed_data.labels[i] not in nanobodies])

        all_mse_without_region.append(np.mean((sums_without_region - train_y_)**2))
    total_mse = all_mse_without_region[-1]
    all_mse_without_region.pop(-1)

    region_mean_lengths = np.array([24, 7.1, 19, 6, 41, 11.3, 10, 21.7, 12.7, 14.9, 7, 32.9, 9.2, 9.1])
    idx_best_normalised_mean_length = np.argmax(abs(all_mse_without_region-total_mse)/region_mean_lengths)

    tot = 100*region_mean_lengths[idx_best_normalised_mean_length] * abs(all_mse_without_region-total_mse) / abs(all_mse_without_region[idx_best_normalised_mean_length]-total_mse)/region_mean_lengths
    ob = tot - tot * abs(all_mse_without_region_intra-total_mse) / (abs(all_mse_without_region_intra-total_mse)+abs(all_mse_without_region_ob-total_mse))
    plot_region_importance(tot, ob, type_of_antigen, mode, interactive=interactive)

def compute_residue_importance(preprocessed_data, model, type_of_antigen, nanobodies, interactive=False):
    r"""Computes the importance factors (0-100) of all the amino acids of the antibody variable region.

    Parameters
    ----------
    preprocessed_data: antipasti.preprocessing.preprocessing.Preprocessing
        The ``Preprocessing`` class.
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    type_of_antigen: int
        Choose between: proteins (0), haptens (1), peptides (2) or carbohydrates (3).
    nanobodies: list
        PDB codes of nanobodies in the dataset.
    interactive: bool
        Set to ``True`` when running a script or ``pytest``.

    """

    colours = get_colours_ag_type(preprocessed_data)
    each_img_enl = get_output_representations(preprocessed_data, model)

    train_x = preprocessed_data.train_x
    input_shape = preprocessed_data.test_x.shape[-1]

    colours = [0 if c == 5 else c for c in colours]
    all_mse_without_region = []

    for j in range(train_x.shape[-1]+1):
        if j != train_x.shape[-1]:
            sums_without_region = np.array([
                    each_img_enl[i].reshape((input_shape, input_shape)).sum()-(each_img_enl[i].reshape((input_shape, input_shape))[j:j+1, :]).sum()
                    for i in range(train_x.shape[0]) if colours[i] == type_of_antigen and preprocessed_data.labels[i] not in nanobodies])
        else:
            sums_without_region = np.array([
                    each_img_enl[i].reshape((input_shape, input_shape)).sum()
                    for i in range(train_x.shape[0]) if colours[i] == type_of_antigen and preprocessed_data.labels[i] not in nanobodies])
        train_y_ = np.array([preprocessed_data.train_y[i] for i in range(train_x.shape[0]) if colours[i] == type_of_antigen and preprocessed_data.labels[i] not in nanobodies])
        
        mse_without_region = np.mean((sums_without_region - train_y_)**2)
        all_mse_without_region.append(mse_without_region)
    total_mse = all_mse_without_region[-1]
    all_mse_without_region.pop(-1)

    plot_residue_importance(preprocessed_data, 100*abs(all_mse_without_region-total_mse)/abs(max(all_mse_without_region)-total_mse), type_of_antigen, interactive=interactive)