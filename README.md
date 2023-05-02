# Predicting and explaining antibody binding affinity with Deep Learning and Normal Mode Analysis

[![Python 3.8 3.9 3.10](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/license/mit/)
[![Tests](https://github.com/kevinmicha/predicting-affinity/actions/workflows/tests.yml/badge.svg)](https://github.com/kevinmicha/predicting-affinity/actions/workflows/tests.yml)
[![Docs](https://github.com/kevinmicha/predicting-affinity/actions/workflows/documentation.yml/badge.svg)](https://kevinmicha.github.io/predicting-affinity/index.html)

NMA-CNN is a Python Deep Learning method that predicts the binding affinity of antibodies from their sequence and three-dimensional structure.

## Installation 
### Through Anaconda
We provide an Anaconda environment that satisfies all the dependencies in `nma-cnn-env.yml`. 
```bash
git clone https://github.com/kevinmicha/predicting-affinity
cd predicting-affinity
conda env create -f nma-cnn-env.yml
conda activate nma-cnn-env
pip install .
```

Next, you can run the tests to make sure your installation is working correctly.

```bash
# While still in the NMA-CNN directory:
pytest . 
```

### Manually handling the dependencies
If you want to use an existing environment, just omit the Anaconda commands above:
```bash
git clone https://github.com/kevinmicha/predicting-affinity
cd predicting-affinity
pip install .
```

or if you need to install it for your user only: 
```bash
python setup.py install --user 
```

## Requirements 

NMA-CNN requires the following Python packages: 
* `adabelief-pytorch`
* `contact-map`
* `matplotlib`
* `mdtraj`
* `numpy`
* `opencv-python`
* `pandas`
* `scikit-image`
* `torch`
* `torchmetrics`
* `tqdm`
    


## Example Notebooks and Documentation
The full documentation can be found [here](https://kevinmicha.github.io/predicting-affinity/). 

Example notebooks are located in the [notebooks](https://github.com/kevinmicha/predicting-affinity/tree/main/notebooks) folder:
* [[Tutorial] Training NMA-CNN with paired HL](https://github.com/kevinmicha/predicting-affinity/blob/main/notebooks/%5BTutorial%5D%20NMA%20CNN%20HL%20paired.ipynb)

## Attribution

If you use this code, please cite the [paper](https://kevinmicha.github.io/predicting-affinity/citing.html) indicated in the documentation.