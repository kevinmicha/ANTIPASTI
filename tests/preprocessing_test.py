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
        self.pathological = ['5omm', '5i5k', '1uwx', '1mj7', '1qfw', '1qyg', '4ffz', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t', '3fku', '1oau', '1oay'] + ['4gqp', '3etb', '3gkz', '3uze', '3uzq', '3gm0', '4f9l', '6ejg', '6ejm', '1h8s', '5dfw', '6cbp', '4f9p', '5kov', '1dzb', '5j74', '5aaw', '3uzv', '5aam', '3ux9', '5a2j', '5a2k', '5a2i', '3fku', '5yy4', '3uyp', '5jyl', '1y0l', '1p4b', '3kdm', '4lar', '4ffy', '2ybr', '1mfa', '5xj3', '5xj4', '4kv5', '5vyf'] 

    def test_new_maps(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=True, renew_residues=True)

    def test_new_lengths(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=False, renew_residues=True)

    def test_load(self):
        preprocessed_data = Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=False, renew_residues=False)

    def test_contact_maps(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=True, renew_residues=False, cmaps=True)

    def test_antigen_agnostic(self):
        Preprocessing(data_path=self.data_path, structures_path=self.structures_path, scripts_path=self.scripts_path, df=self.df, pathological=self.pathological, renew_maps=True, renew_residues=True, ag_agnostic=True)