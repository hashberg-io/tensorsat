Getting Started
===============

TensorSat is an experimental project to explore the usage of hyper-optimised tensor network contraction techniques in satisfiability (SAT/SMT solving).

Installation
------------

You can install the latest release from `PyPI <https://pypi.org/project/tensorsat/>`_ as follows:

.. code-block:: console

    $ pip install --upgrade tensorsat

For visualization, you should additionally install `matplotlib <https://matplotlib.org/>`_ and `networkx <https://networkx.org/>`_:

.. code-block:: console

    $ pip install --upgrade matplotlib networkx

For optimised tensor contraction, you should additionally install `cotengra <https://cotengra.readthedocs.io/en/latest/>`_:

.. code-block:: console

    $ pip install --upgrade cotengra

The following are the recommended optional dependencies for Cotengra:

.. code-block:: console

    $ pip install --upgrade kahypar autoray cmaes cotengrust cytoolz loky networkx opt_einsum optuna tqdm

Please note that `kahypar <https://github.com/kahypar/kahypar>`_ is not available for Windows on PyPI: If you wish to build it on a Windows machine, please follow the instructions at https://github.com/kahypar/kahypar#the-python-interface.


GitHub repo: https://github.com/hashberg-io/tensorsat
