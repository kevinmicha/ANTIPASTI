import os
import numpy as np
import pytest
import torch
import unittest

from antipasti.preprocessing.preprocessing import Preprocessing
from antipasti.utils.torch_utils import load_checkpoint

from tests import TEST_PATH

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.modes = 'all'
        self.n_filters = 4
        self.filter_size = 4
        self.pooling_size = 1
        self.n_max_epochs = 935
        self.df = 'summary.tsv'
        self.test_data_path = os.path.join('notebooks/', 'test_data/')
        self.test_dccm_map_path = 'dccm_map/'
        self.test_residues_path = 'list_of_residues/'
        self.test_structure_path = 'structure/'
        self.scripts_path = './scripts/'
        self.stage = 'predicting'
        self.pathological = ['5omm', '5i5k', '1uwx', '1mj7', '1qfw', '1qyg', '4ffz', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t', '3fku', '1oau', '1oay'] + ['4gqp', '3etb', '3gkz', '3uze', '3uzq', '3gm0', '4f9l', '6ejg', '6ejm', '1h8s', '5dfw', '6cbp', '4f9p', '5kov', '1dzb', '5j74', '5aaw', '3uzv', '5aam', '3ux9', '5a2j', '5a2k', '5a2i', '3fku', '5yy4', '3uyp', '5jyl', '1y0l', '1p4b', '3kdm', '4lar', '4ffy', '2ybr', '1mfa', '5xj3', '5xj4', '4kv5', '5vyf'] 
        self.dccm_map_path = 'dccm_maps_full_ags_all/'

    def test_evaluation(self):

        preprocessed_data = Preprocessing(data_path='data/', dccm_map_path=self.dccm_map_path, modes=self.modes, pathological=self.pathological, stage=self.stage, test_data_path=self.test_data_path, test_dccm_map_path=self.test_dccm_map_path, test_residues_path=self.test_residues_path, test_structure_path=self.test_structure_path, renew_residues=True)
        input_shape = preprocessed_data.test_x.shape[-1]

        # Testing cases of load checkpoint
        path = 'checkpoints/full_ags_all_modes/model_epochs_' + str(self.n_max_epochs) + '_modes_' + str(self.modes) + '_pool_' + str(self.pooling_size) + '_filters_' + str(self.n_filters) + '_size_' + str(self.filter_size) + '.pt'
        load_checkpoint(path, input_shape)
        load_checkpoint(path, input_shape, n_filters=self.n_filters, pooling_size=self.pooling_size, filter_size=self.filter_size)

        # Same idea with checkpoint originated from a reduced amount of data
        self.data_path = os.path.join(self.path, 'data/')
        preprocessed_data = Preprocessing(data_path=self.data_path, test_data_path=self.test_data_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, stage=self.stage, test_dccm_map_path=self.test_dccm_map_path, test_residues_path=self.test_residues_path, test_structure_path=self.test_structure_path, renew_residues=True)
        input_shape = preprocessed_data.test_x.shape[-1]

        path = 'checkpoints/model_unit_test.pt'
        model = load_checkpoint(path, input_shape)[0]
        model.eval()

        test_sample = torch.from_numpy(preprocessed_data.test_x.reshape(1, 1, input_shape, input_shape).astype(np.float32))
        model(test_sample)