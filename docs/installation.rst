Installation
============


Requirements
------------

NMA-CNN is developed using Python 3.8, 3.9 and 3.10 and might be fine with older versions.

It requires ``adabelief-pytorch``, ``contact-map``, ``matplotlib``, ``mdtraj``, ``numpy``, ``opencv-python``, ``pandas``, ``scikit-learn``, ``torch``, ``torchmetrics`` and ``tqdm`` to work properly. 

Through Anaconda 
----------------

We provide an Anaconda environment that satisfies all the dependencies in ``nma-cnn-env.yml``.

::

    git clone https://github.com/kevinmicha/predicting-affinity
    cd predicting-affinity
    conda env create -f nma-cnn-env.yml
    conda activate nma-cnn-env
    pip install .

Next, you can run the tests to make sure your installation is working correctly.

::

    # While still in the NMA-CNN directory:
    pytest . 

    
Manually handling the dependencies
----------------------------------

If you want to use an existing environment, just omit the Anaconda commands above:
::

    git clone https://github.com/kevinmicha/predicting-affinity
    cd predicting-affinity
    pip install .


or if you need to install it for your user only:

::

	python setup.py install --user 
