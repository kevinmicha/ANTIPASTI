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
        self.modes = 30
        self.n_filters = 2
        self.filter_size = 4
        self.pooling_size = 1
        self.n_max_epochs = 552
        self.data_path = 'data/'
        self.regions = 'paired_hl'
        self.df = 'summary.tsv'
        self.test_data_path = os.path.join('notebooks/', 'test_data/')
        self.test_dccm_map_path = 'dccm_map/'
        self.test_residues_path = 'list_of_residues/'
        self.test_structure_path = 'structure/'
        self.scripts_path = 'scripts/'
        self.stage = 'predicting'
        self.pathological = ['5omm', '1mj7', '1qfw', '1qyg', '4ffz', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t']

    def test_evaluation(self):

        preprocessed_data = Preprocessing(data_path=self.data_path, modes=self.modes, pathological=self.pathological, regions=self.regions, stage=self.stage, test_data_path=self.test_data_path, test_dccm_map_path=self.test_dccm_map_path, test_residues_path=self.test_residues_path, test_structure_path=self.test_structure_path)
        input_shape = preprocessed_data.test_x.shape[-1]

        # Testing cases of load checkpoint
        path = 'checkpoints/model_' + str(self.regions) + '_epochs_' + str(self.n_max_epochs) + '_modes_' + str(self.modes) + '_pool_' + str(self.pooling_size) + '_filters_' + str(self.n_filters) + '_size_' + str(self.filter_size) + '.pt'
        load_checkpoint(path, input_shape)
        load_checkpoint(path, input_shape, n_filters=self.n_filters, pooling_size=self.pooling_size, filter_size=self.filter_size)

        # Same idea with checkpoint originated from a reduced amount of data
        self.data_path = os.path.join(self.path, 'data/')
        self.structures_path = self.data_path + 'structures/'
        preprocessed_data = Preprocessing(data_path=self.data_path, test_data_path=self.test_data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, stage=self.stage, test_dccm_map_path=self.test_dccm_map_path, test_residues_path=self.test_residues_path, test_structure_path=self.test_structure_path)
        input_shape = preprocessed_data.test_x.shape[-1]

        path = 'checkpoints/model_unit_test.pt'
        model = load_checkpoint(path, input_shape)[0]
        model.eval()

        test_sample = torch.from_numpy(preprocessed_data.test_x.reshape(1, 1, input_shape, input_shape).astype(np.float32))
        model(test_sample)