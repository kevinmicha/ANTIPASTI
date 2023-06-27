import numpy as np
import os
import pytest
import unittest

# ANTIPASTI
from antipasti.preprocessing.preprocessing import Preprocessing
from antipasti.utils.explaining_utils import compute_change_in_kd, get_epsilon, get_maps_of_interest, get_test_contribution, map_residues_to_regions, plot_map_with_regions
from antipasti.utils.torch_utils import load_checkpoint
from tests import TEST_PATH

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.modes = 30
        self.n_filters = 2
        self.filter_size = 4
        self.pooling_size = 1
        self.n_max_epochs = 336

        self.mode = 'fully-extended' # Choose between 'fully-extended' and 'fully-cropped'
        self.pathological = ['3etb', '3gkz', '3lrh', '3t0w', '3t0x', '3uze', '3uzq', '4f9l', '4gqp', '4k3h', '6d6t']
        self.stage = 'predicting'
        self.regions = 'paired_hl'
        self.data_path = 'data/'
        self.test_data_path = os.path.join('notebooks/', 'test_data/')
        self.test_dccm_map_path = 'dccm_map/'
        self.test_residues_path = 'list_of_residues/'
        self.test_structure_path = 'structure/'
        
    def test_explaining(self):

        # Pre-processing
        preprocessed_data = Preprocessing(data_path=self.data_path, modes=self.modes, pathological=self.pathological, regions=self.regions, mode=self.mode, stage=self.stage, test_data_path=self.test_data_path, test_dccm_map_path=self.test_dccm_map_path, test_residues_path=self.test_residues_path, test_structure_path=self.test_structure_path)
        input_shape = preprocessed_data.test_x.shape[-1]
        print(input_shape)
        # Loading the actual checkpoint and learnt filters
        path = 'checkpoints/model_' + self.regions + '_epochs_' + str(self.n_max_epochs) + '_modes_' + str(self.modes) + '_pool_' + str(self.pooling_size) + '_filters_' + str(self.n_filters) + '_size_' + str(self.filter_size) + '.pt'
        model = load_checkpoint(path, input_shape)[0]
        learnt_filter = np.load('checkpoints/learnt_filter_'+self.regions+'_epochs_'+str(self.n_max_epochs)+'_modes_'+str(self.modes)+'_pool_'+str(self.pooling_size)+'_filters_'+str(self.n_filters)+'_size_'+str(self.filter_size)+'.npy')
        model.eval()

        mean_learnt, mean_image, mean_diff_image = get_maps_of_interest(preprocessed_data, learnt_filter)
        plot_map_with_regions(preprocessed_data, mean_learnt, 'Average of learnt representations', True)
        
        contribution = get_test_contribution(preprocessed_data, model)
        epsilon = get_epsilon(preprocessed_data, model)
        epsilon = get_epsilon(preprocessed_data, model, mode='extreme')
        coord, maps, labels = map_residues_to_regions(preprocessed_data, epsilon)
        get_test_contribution(preprocessed_data, model)
        
        # Expressing weights as vector
        weights_h = [0.1, 0.1, 0, 0.1, 0, 0.1, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0.1]
        weights_l = [0.1, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0.1]
        weights = np.array(weights_h + weights_l)

        compute_change_in_kd(preprocessed_data, model, weights, coord, maps)