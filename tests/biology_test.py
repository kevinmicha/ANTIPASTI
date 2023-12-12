import pytest
import unittest

from antipasti.utils.biology_utils import extract_mean_region_lengths

from tests import TEST_PATH

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH

    def test_biology(self):
        extract_mean_region_lengths(['1fl6'], data_path=self.path+'/data/')
        extract_mean_region_lengths(['5d70'], data_path='data/')
