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

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

setup(
    name='ANTIPASTI',
    version='1.1',
    author='Kevin Michalewicz',
    author_email='k.michalewicz22@imperial.ac.uk',
    description='Deep Learning model that predicts the binding affinity of antibodies from their three-dimensional structure.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['antipasti', 'antipasti.model', 'antipasti.preprocessing', 'antipasti.utils'],
    install_requires=['adabelief-pytorch', 'biopython', 'matplotlib', 'numpy', 'opencv-python', 'optuna', 'pandas', 'requests', 'scikit-learn', 'scipy', 'torch', 'torchmetrics', 'umap-learn'],
    cmdclass={'test': PyTest}
)
