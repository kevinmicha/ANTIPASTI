import os
import pytest
import unittest

from adabelief_pytorch import AdaBelief
from torch.nn import MSELoss

from nmacnn.model.model import NormalModeAnalysisCNN
from nmacnn.preprocessing.preprocessing import Preprocessing
from nmacnn.utils.torch_utils import create_validation_set, training_routine
from tests import TEST_PATH

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.data_path = os.path.join(self.path, 'data/')
        self.structures_path = self.data_path + 'structures/'
        self.scripts_path = './scripts/'
        self.df = 'summary.tsv'
        self.pathological = ['5omm', '1mj7', '1qfw', '1qyg', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t']

    def test_training(self):
        preprocessed_data = Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=False, renew_residues=False)
        train_x, val_x, train_y, val_y = create_validation_set(preprocessed_data.train_x, preprocessed_data.train_y, val_size=0.5)

        n_filters = 2
        filter_size = 5
        pooling_size = 1
        learning_rate = 4e-4
        n_max_epochs = 5
        max_corr = 0.87
        batch_size = 1
        input_shape = preprocessed_data.train_x.shape[-1]

        model = NormalModeAnalysisCNN(n_filters=n_filters, filter_size=filter_size, pooling_size=pooling_size, input_shape=input_shape)
        criterion = MSELoss()
        optimiser = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-8, print_change_log=False) 

        training_routine(model, criterion, optimiser, train_x, val_x, train_y, val_y, n_max_epochs=n_max_epochs, max_corr=max_corr, batch_size=batch_size, verbose=False)

if __name__ == '__main__':
    pytest.main()