import pytest
import unittest

from antipasti.utils.biology_utils import get_cdr_lengths, get_types_of_residues

from tests import TEST_PATH

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH

    def test_biology(self):
        get_cdr_lengths(['1fl6', '4fab', '5d70', '1kxt', '1g6v', '2p44', '2jb6', '6b9j'], 'data/')
        get_types_of_residues(['1t66', '1kel', '6mlb', 'abcd'])
