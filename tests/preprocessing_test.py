import os
import pytest
import unittest

from nmacnn.preprocessing.preprocessing import Preprocessing
from tests import TEST_PATH

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.data_path = os.path.join(self.path, 'data/')
        self.structures_path = self.data_path + 'structures/'
        self.scripts_path = './scripts/'
        self.df = '28B4_summary.tsv'
        self.pathological = ['5omm', '1mj7', '1qfw', '1qyg', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t']

    def test_new_maps(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=True, renew_residues=False)

    def test_new_lengths(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=False, renew_residues=True)

    def test_load(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=False, renew_residues=False)

if __name__ == '__main__':
    pytest.main()
