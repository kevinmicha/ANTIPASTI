import sys
from distutils.core import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(
    name='NMA-CNN',
    version='1.0',
    author='Kevin Michalewicz',
    author_email='k.michalewicz22@imperial.ac.uk',
    description='Deep Learning model that predicts the binding affinity of antibodies from their sequence and three-dimensional structure.',
    packages=['nma-cnn', 'nma-cnn.model', 'nma-cnn.plots', 'nma-cnn.preprocessing', 'nma-cnn.utils'],
    requires=['adabelief-pytorch', 'contact-map', 'matplotlib', 'mdtraj', 'numpy', 'opencv-python', 'pandas', 'scikit-learn', 'torch', 'torchmetrics', 'tqdm'],
    cmdclass={'test': PyTest}
)
