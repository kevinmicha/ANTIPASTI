import pytest
import unittest

from antipasti.utils.biology_utils import extract_mean_region_lengths

from tests import TEST_PATH

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH

    def test_biology(self):
        extract_mean_region_lengths(['1fl6', '1mlc'], data_path=self.path+'/data/')
        extract_mean_region_lengths(['5d70', '1g6v', '6bkb', '5boz', '2jb6'], data_path='data/') # Problems in CDR-L2, CDR-H2, CDR-H1 (x3) resp.

