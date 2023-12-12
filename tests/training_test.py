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
        self.pathological = ['5omm', '5i5k', '1uwx', '1mj7', '1qfw', '1qyg', '4ffz', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t', '3fku', '1oau', '1oay'] + ['4gqp', '3etb', '3gkz', '3uze', '3uzq', '3gm0', '4f9l', '6ejg', '6ejm', '1h8s', '5dfw', '6cbp', '4f9p', '5kov', '1dzb', '5j74', '5aaw', '3uzv', '5aam', '3ux9', '5a2j', '5a2k', '5a2i', '3fku', '5yy4', '3uyp', '5jyl', '1y0l', '1p4b', '3kdm', '4lar', '4ffy', '2ybr', '1mfa', '5xj3', '5xj4', '4kv5', '5vyf'] 
    
    def test_training_paired_hl(self):
        preprocessed_data = Preprocessing(data_path='data/', dccm_map_path='dccm_maps_full_ags_all/', pathological=self.pathological, renew_residues=True)
        train_x, test_x, train_y, test_y, _, _ = create_test_set(preprocessed_data.train_x, preprocessed_data.train_y)

        # Small training set
        preprocessed_data = Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological)
        train_x, test_x, train_y, test_y, _, _ = create_test_set(preprocessed_data.train_x, preprocessed_data.train_y, test_size=0.5)

        n_filters = 2
        filter_size = 4
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
        train_loss, test_loss, _, _, _ = training_routine(model, criterion, optimiser, train_x, test_x, train_y, test_y, n_max_epochs=n_max_epochs, max_corr=max_corr, batch_size=batch_size, verbose=True)

        # Saving the losses
        train_losses.extend(train_loss)
        test_losses.extend(test_loss)

        # Saving the checkpoint
        path = 'checkpoints/model_unit_test.pt'
        save_checkpoint(path, model, optimiser, train_losses, test_losses)