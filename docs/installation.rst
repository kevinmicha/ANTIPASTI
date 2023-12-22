Installation
============


Requirements
------------

ANTIPASTI is developed using Python 3.8, 3.9, 3.10 and 3.11 and might be fine with older versions.

It requires ``adabelief-pytorch``, ``biopython``, ``matplotlib``, ``numpy``, ``opencv-python``, ``optuna``, ``pandas``, ``scikit-learn``, ``scipy``, ``torch``, and ``torchmetrics`` to work properly. 

Through PyPI
------------

ANTIPASTI releases are distributed through the Python Package Index (PyPI). To install the latest version use ``pip``:

::

    $ pip install antipasti

   
Through Anaconda 
----------------

We provide an Anaconda environment that satisfies all the dependencies in ``antipasti-env.yml``.

::

    git clone https://github.com/kevinmicha/ANTIPASTI
    cd ANTIPASTI
    conda env create -f antipasti-env.yml
    conda activate antipasti-env
    pip install .

Next, you can run the tests to make sure your installation is working correctly.

::

    # While still in the ANTIPASTI directory:
    pytest . 

    
Manually handling the dependencies
----------------------------------

If you want to use an existing environment, just omit the Anaconda commands above:
::

    git clone https://github.com/kevinmicha/ANTIPASTI
    cd ANTIPASTI
    pip install .


or if you need to install it for your user only:

::

	python setup.py install --user 
