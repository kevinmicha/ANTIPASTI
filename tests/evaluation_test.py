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
        self.data_path = os.path.join(self.path, 'data/')
        self.df = 'summary.tsv'
        self.test_data_path = os.path.join('notebooks/', 'test_data/')
        self.structures_path = self.data_path + 'structures/'
        self.test_dccm_map_path = 'dccm_map/'
        self.test_residues_path = 'list_of_residues/'
        self.test_structure_path = 'structure/'
        self.scripts_path = 'scripts/'
        self.stage = 'predicting'
        self.pathological = ['3etb', '3gkz', '3lrh', '3t0w', '3t0x', '3uze', '3uzq', '4f9l', '4gqp', '4k3h', '6d6t']

    def test_evaluation(self):

        preprocessed_data = Preprocessing(data_path=self.data_path, test_data_path=self.test_data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, stage=self.stage, test_dccm_map_path=self.test_dccm_map_path, test_residues_path=self.test_residues_path, test_structure_path=self.test_structure_path)
        input_shape = preprocessed_data.test_x.shape[-1]

        # Testing cases of load checkpoint
        path = 'checkpoints/model_unit_test_modes_30_pool_1_filters_2_size_5.pt'
        load_checkpoint(path, input_shape)
        load_checkpoint(path, input_shape, n_filters=2, pooling_size=1, filter_size=5)

        path = 'checkpoints/model_unit_test.pt'
        model = load_checkpoint(path, input_shape)[0]
        model.eval()

        test_sample = torch.from_numpy(preprocessed_data.test_x.reshape(1, 1, input_shape, input_shape).astype(np.float32))
        model(test_sample)