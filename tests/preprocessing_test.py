import os
import pytest
import unittest

from antipasti.preprocessing.preprocessing import Preprocessing
from tests import TEST_PATH

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.data_path = os.path.join(self.path, 'data/')
        self.structures_path = self.data_path + 'structures/'
        self.scripts_path = './scripts/'
        self.df = 'summary.tsv'
        self.pathological = ['3etb', '3gkz', '3lrh', '3t0w', '3t0x', '3uze', '3uzq', '4f9l', '4gqp', '4k3h', '6d6t']
        self.regions = 'heavy'

    def test_new_maps_heavy(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, regions=self.regions, pathological=self.pathological, renew_maps=True, renew_residues=False)

    def test_new_lengths_heavy(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, regions=self.regions, pathological=self.pathological, renew_maps=False, renew_residues=True)

    def test_load_heavy(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, regions=self.regions, pathological=self.pathological, renew_maps=False, renew_residues=False)

    def test_new_maps(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=True, renew_residues=False)

    def test_new_lengths(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=False, renew_residues=True)

    def test_load(self):
        preprocessed_data = Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=False, renew_residues=False)