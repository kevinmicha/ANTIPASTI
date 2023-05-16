import os
import pytest
import unittest

from adabelief_pytorch import AdaBelief
from torch.nn import MSELoss

from antipasti.model.model import ANTIPASTI
from antipasti.preprocessing.preprocessing import Preprocessing
from antipasti.utils.torch_utils import create_test_set, save_checkpoint, training_routine
from antipasti.config import CHECKPOINTS_DIR
from tests import TEST_PATH

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.data_path = os.path.join(self.path, 'data/')
        self.structures_path = self.data_path + 'structures/'
        self.scripts_path = './scripts/'
        self.df = 'summary.tsv'
        self.pathological = ['5omm', '1mj7', '1qfw', '1qyg', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t']

    def test_training_paired_hl(self):
        preprocessed_data = Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological)
        train_x, test_x, train_y, test_y = create_test_set(preprocessed_data.train_x, preprocessed_data.train_y, test_size=0.5)

        n_filters = 2
        filter_size = 5
        pooling_size = 1
        learning_rate = 4e-4
        n_max_epochs = 10
        max_corr = 0.87
        batch_size = 1
        input_shape = preprocessed_data.train_x.shape[-1]

        model = ANTIPASTI(n_filters=n_filters, filter_size=filter_size, pooling_size=pooling_size, input_shape=input_shape)
        criterion = MSELoss()
        optimiser = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-8, print_change_log=False) 

        train_losses = []
        test_losses = []
        train_loss, test_loss, _, _, _ = training_routine(model, criterion, optimiser, train_x, test_x, train_y, test_y, n_max_epochs=n_max_epochs, max_corr=max_corr, batch_size=batch_size, verbose=False)

        # Saving the losses
        train_losses.extend(train_loss)
        test_losses.extend(test_loss)

        # Saving the checkpoint
        path = 'checkpoints/model_unit_test.pt'
        save_checkpoint(path, model, optimiser, train_losses, test_losses)

    def test_training_heavy(self):
        preprocessed_data = Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, regions='heavy', pathological=self.pathological)
        train_x, test_x, train_y, test_y = create_test_set(preprocessed_data.train_x, preprocessed_data.train_y, test_size=0.5)

        n_filters = 3
        filter_size = 5
        pooling_size = 2
        learning_rate = 4e-4
        n_max_epochs = 10
        max_corr = 0.87
        batch_size = 1
        input_shape = preprocessed_data.train_x.shape[-1]

        model = ANTIPASTI(n_filters=n_filters, filter_size=filter_size, pooling_size=pooling_size, input_shape=input_shape)
        criterion = MSELoss()
        optimiser = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-8, print_change_log=False) 

        train_losses = []
        test_losses = []
        train_loss, test_loss, _, _, _ = training_routine(model, criterion, optimiser, train_x, test_x, train_y, test_y, n_max_epochs=n_max_epochs, max_corr=max_corr, batch_size=batch_size, verbose=False)

        # Saving the losses
        train_losses.extend(train_loss)
        test_losses.extend(test_loss)

if __name__ == '__main__':
    pytest.main()