import numpy as np
import os
import pytest
import unittest

# ANTIPASTI
from antipasti.preprocessing.preprocessing import Preprocessing
from antipasti.utils.explaining_utils import compute_umap, compute_region_importance, compute_residue_importance, get_maps_of_interest, get_test_contribution, plot_map_with_regions
from antipasti.utils.torch_utils import load_checkpoint
from tests import TEST_PATH

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.modes = 100
        self.n_filters = 4
        self.filter_size = 4
        self.pooling_size = 1
        self.n_max_epochs = 422

        self.pathological = ['5omm', '5i5k', '1uwx', '1mj7', '1qfw', '1qyg', '4ffz', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t', '3fku', '1oau', '1oay'] + ['4gqp', '3etb', '3gkz', '3uze', '3uzq', '3gm0', '4f9l', '6ejg', '6ejm', '1h8s', '5dfw', '6cbp', '4f9p', '5kov', '1dzb', '5j74', '5aaw', '3uzv', '5aam', '3ux9', '5a2j', '5a2k', '5a2i', '3fku', '5yy4', '3uyp', '5jyl', '1y0l', '1p4b', '3kdm', '4lar', '4ffy', '2ybr', '1mfa', '5xj3', '5xj4', '4kv5', '5vyf'] 
        self.nanobodies = ['1g6v', '1kxq', '1kxt', '1kxv', '1op9', '1ri8', '1zmy', '1zv5', '1zvy', '2p42', '2p43', '2p44', '2p45', '2p46', '2p47', '2p48', '2p49', '2p4a', '2vyr', '2x89', '3eba', '3ogo', '3p0g', '3p9w', '3qsk', '3qxt', '3qxv', '3zkq', '4eig', '4eiz', '4hjj', '4kfz', '4krl', '4pgj', '4pou', '4w6w', '4w6y', '4wem', '4wen', '4weu', '4x7d', '4x7e', '4x7f', '4y8d', '4z9k', '5boz', '5dmj', '5e7f', '5fhx', '5fv2', '5hgg', '5hvf', '5hvg', '5imk', '5imm', '5ip4', '5ivn', '5j56', '5j57', '5jds', '5lhn', '5m2i', '5m2j', '5m2m', '5my6', '5mzv', '5n88', '5nqw', '5o03', '5o05', '5o0w', '5o2u', '5omm', '5omn', '5sv3', '5toj', '5u4m', '5vm0', '5y7z', '5y80', '6ehg', '6fe4', '6h7n', '6h7o']
        self.stage = 'predicting'
        self.test_data_path = os.path.join('notebooks/', 'test_data/')
        self.test_dccm_map_path = 'dccm_map/'
        self.test_residues_path = 'list_of_residues/'
        self.test_structure_path = 'structure/'
        self.dccm_map_path = 'dccm_maps_full_ags_100/'
        
    def test_explaining(self):

        # Example class dictionary
        cdict = {'homo sapiens': 0,
            'mus musculus': 1,
            'Other': 2}
        
        # Pre-processing
        preprocessed_data = Preprocessing(data_path='data/', dccm_map_path=self.dccm_map_path, modes=self.modes, pathological=self.pathological, stage=self.stage, test_data_path=self.test_data_path, test_dccm_map_path=self.test_dccm_map_path, test_residues_path=self.test_residues_path, test_structure_path=self.test_structure_path, test_pdb_id='4yhi', residues_path='lists_of_residues/', renew_residues=True)
        input_shape = preprocessed_data.test_x.shape[-1]

        # Loading the actual checkpoint and learnt filters
        path = 'checkpoints/full_ags_n_modes/100_modes/model_epochs_' + str(self.n_max_epochs) + '_modes_' + str(self.modes) + '_pool_' + str(self.pooling_size) + '_filters_' + str(self.n_filters) + '_size_' + str(self.filter_size) + '.pt'
        model = load_checkpoint(path, input_shape)[0]
        learnt_filter = np.load('checkpoints/full_ags_n_modes/100_modes/learnt_filter_epochs_'+str(self.n_max_epochs)+'_modes_'+str(self.modes)+'_pool_'+str(self.pooling_size)+'_filters_'+str(self.n_filters)+'_size_'+str(self.filter_size)+'.npy')
        model.eval()

        mean_learnt, mean_image, mean_diff_image = get_maps_of_interest(preprocessed_data, learnt_filter)
        plot_map_with_regions(preprocessed_data, mean_learnt, 'Average of learnt representations', True)
        get_test_contribution(preprocessed_data, model)
        
        random_sequence = list(np.linspace(0, 1, num=preprocessed_data.train_x.shape[0]))
        random_sequence_delete = random_sequence.copy()
        random_sequence_delete[0] = 'unknown'
        random_sequence = [str(random_sequence[0])] + random_sequence[1:]

        # UMAP cases
        compute_umap(preprocessed_data, model, scheme='heavy_species', categorical=True, include_ellipses=True, numerical_values=None, external_cdict=None, interactive=True, exclude_nanobodies=True)
        compute_umap(preprocessed_data, model, scheme='heavy_species', categorical=True, include_ellipses=True, numerical_values=None, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='light_ctype', categorical=True, include_ellipses=True, numerical_values=None, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='heavy_subclass', categorical=True, include_ellipses=False, numerical_values=None, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='light_subclass', categorical=True, include_ellipses=False, numerical_values=None, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='antigen_type', categorical=True, include_ellipses=False, numerical_values=None, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='antigen_species', categorical=True, include_ellipses=False, numerical_values=None, external_cdict=cdict, interactive=True)
        compute_umap(preprocessed_data, model, scheme='Random sequence', categorical=False, include_ellipses=False, numerical_values=random_sequence, external_cdict=None, interactive=True)
        compute_umap(preprocessed_data, model, scheme='Random sequence', categorical=False, include_ellipses=False, numerical_values=random_sequence, external_cdict=None, interactive=True, exclude_nanobodies=True)
        compute_umap(preprocessed_data, model, scheme='Random sequence', categorical=False, include_ellipses=False, numerical_values=random_sequence_delete, external_cdict=None, interactive=True)

        # Region and residue importance cases
        compute_region_importance(preprocessed_data, model, 0, self.nanobodies, mode='region', interactive=True)
        compute_region_importance(preprocessed_data, model, 3, self.nanobodies, mode='chain', interactive=True)
        compute_residue_importance(preprocessed_data, model, 2, self.nanobodies, interactive=True)
