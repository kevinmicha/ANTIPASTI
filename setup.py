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
    name='ANTIPASTI',
    version='1.0',
    author='Kevin Michalewicz',
    author_email='k.michalewicz22@imperial.ac.uk',
    description='Deep Learning model that predicts the binding affinity of antibodies from their sequence and three-dimensional structure.',
    packages=['antipasti', 'antipasti.model', 'antipasti.preprocessing', 'antipasti.utils'],
    install_requires=['adabelief-pytorch', 'beautifulsoup4', 'matplotlib', 'numpy', 'opencv-python', 'pandas', 'scikit-learn', 'torch', 'torchmetrics', 'umap-learn'],
    cmdclass={'test': PyTest}
)
