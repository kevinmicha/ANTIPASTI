import numpy as np
import os
import pytest
import unittest

# ANTIPASTI
from antipasti.preprocessing.preprocessing import Preprocessing
from antipasti.utils.explaining_utils import compute_umap, get_maps_of_interest, get_test_contribution, plot_map_with_regions
from antipasti.utils.torch_utils import load_checkpoint
from tests import TEST_PATH

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.modes = 30
        self.n_filters = 2
        self.filter_size = 4
        self.pooling_size = 1
        self.n_max_epochs = 552

        self.mode = 'fully-extended' # Choose between 'fully-extended' and 'fully-cropped'
        self.pathological = ['5omm', '1mj7', '1qfw', '1qyg', '4ffz', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t']
        self.stage = 'predicting'
        self.data_path = 'data/'
        self.test_data_path = os.path.join('notebooks/', 'test_data/')
        self.test_dccm_map_path = 'dccm_map/'
        self.test_residues_path = 'list_of_residues/'
        self.test_structure_path = 'structure/'
        
    def test_explaining(self):

        # Example class dictionary
        cdict = {'homo sapiens': 0,
            'mus musculus': 1,
            'Other': 2}
        
        # Pre-processing
        preprocessed_data = Preprocessing(data_path=self.data_path, modes=self.modes, pathological=self.pathological, mode=self.mode, stage=self.stage, test_data_path=self.test_data_path, test_dccm_map_path=self.test_dccm_map_path, test_residues_path=self.test_residues_path, test_structure_path=self.test_structure_path)
        input_shape = preprocessed_data.test_x.shape[-1]

        # Loading the actual checkpoint and learnt filters
        path = 'checkpoints/model_epochs_' + str(self.n_max_epochs) + '_modes_' + str(self.modes) + '_pool_' + str(self.pooling_size) + '_filters_' + str(self.n_filters) + '_size_' + str(self.filter_size) + '.pt'
        model = load_checkpoint(path, input_shape)[0]
        learnt_filter = np.load('checkpoints/learnt_filter_epochs_'+str(self.n_max_epochs)+'_modes_'+str(self.modes)+'_pool_'+str(self.pooling_size)+'_filters_'+str(self.n_filters)+'_size_'+str(self.filter_size)+'.npy')
        model.eval()

        mean_learnt, mean_image, mean_diff_image = get_maps_of_interest(preprocessed_data, learnt_filter)
        plot_map_with_regions(preprocessed_data, mean_learnt, 'Average of learnt representations', True)
        
        get_test_contribution(preprocessed_data, model)
        
        random_sequence = list(np.linspace(0, 1, num=preprocessed_data.train_x.shape[0]))
        random_sequence_delete = random_sequence.copy()
        random_sequence_delete[0] = 'unknown'
        random_sequence = [str(random_sequence[0])] + random_sequence[1:]

        compute_umap(preprocessed_data, model, scheme='heavy_species', categorical=True, include_ellipses=True, numerical_values=None, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='light_ctype', categorical=True, include_ellipses=True, numerical_values=None, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='light_subclass', categorical=True, include_ellipses=False, numerical_values=None, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='antigen_type', categorical=True, include_ellipses=False, numerical_values=None, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='antigen_species', categorical=True, include_ellipses=False, numerical_values=None, external_cdict=cdict, interactive=True)
        compute_umap(preprocessed_data, model, scheme='Random sequence', categorical=False, include_ellipses=False, numerical_values=random_sequence, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='Random sequence', categorical=False, include_ellipses=False, numerical_values=random_sequence_delete, external_cdict=None, interactive=True)