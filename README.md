# Predicting and explaining antibody binding affinity with Deep Learning and Normal Mode Analysis

[![Python 3.8 - 3.11](https://img.shields.io/badge/Python-3.8%20--%203.11-blue)](https://www.python.org/downloads/release/python-3113/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/license/mit/)
[![Tests](https://github.com/kevinmicha/ANTIPASTI/actions/workflows/tests.yml/badge.svg)](https://github.com/kevinmicha/ANTIPASTI/actions/workflows/tests.yml)
[![Docs](https://github.com/kevinmicha/ANTIPASTI/actions/workflows/documentation.yml/badge.svg)](https://kevinmicha.github.io/ANTIPASTI/index.html)

ANTIPASTI (ANTIbodies - Predicting Affinity from STructural Information) is a Python Deep Learning method that predicts the binding affinity of antibodies from their sequence and three-dimensional structure.

## Installation 
### Through Anaconda
We provide an Anaconda environment that satisfies all the dependencies in `antipasti-env.yml`. 
```bash
git clone https://github.com/kevinmicha/ANTIPASTI
cd ANTIPASTI
conda env create -f antipasti-env.yml
conda activate antipasti-env
pip install .
```

Next, you can run the tests to make sure your installation is working correctly.

```bash
# While still in the ANTIPASTI directory:
pytest . 
```

### Manually handling the dependencies
If you want to use an existing environment, just omit the Anaconda commands above:
```bash
git clone https://github.com/kevinmicha/ANTIPASTI
cd ANTIPASTI
pip install .
```

or if you need to install it for your user only: 
```bash
python setup.py install --user 
```

## Requirements 

ANTIPASTI requires the following Python packages: 
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
The full documentation can be found [here](https://kevinmicha.github.io/ANTIPASTI/). 

Example notebooks are located in the [notebooks](https://github.com/kevinmicha/ANTIPASTI/tree/main/notebooks) folder:
* [[Tutorial] Training ANTIPASTI](https://github.com/kevinmicha/ANTIPASTI/blob/main/notebooks/%5BTutorial%5D%20Training%20ANTIPASTI.ipynb)
* [[Tutorial] Predicting affinity using ANTIPASTI](https://github.com/kevinmicha/ANTIPASTI/blob/main/notebooks/%5BTutorial%5D%20Predicting%20affinity%20using%20ANTIPASTI.ipynb)
* [[Tutorial] Explaining affinity using ANTIPASTI](https://github.com/kevinmicha/ANTIPASTI/blob/main/notebooks/%5BTutorial%5D%20Explaining%20affinity%20using%20ANTIPASTI.ipynb)
* [[Analysis] AlphaFold can be useful if only sequences are available](https://github.com/kevinmicha/ANTIPASTI/blob/main/notebooks/%5BAnalysis%5D%20AlphaFold%20can%20be%20useful%20if%20only%20sequences%20are%20available.ipynb)

## Attribution

If you use this code, please cite the [paper](https://kevinmicha.github.io/ANTIPASTI/citing.html) indicated in the documentation.