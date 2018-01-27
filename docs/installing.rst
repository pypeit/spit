.. highlight:: rest

***************
Installing spit
***************

This document describes how to install the `spit`
repository.

Installing Dependencies
=======================
There are a number of non-standard dependencies
related to the CNN(s) that lie within SPIT.

In general, we recommend that you use Anaconda for the majority of
these installations.

Detailed installation instructions are presented below:

Python Dependencies
-------------------

specdb depends on the following list of Python packages.

We recommend that you use `Anaconda <https://www.continuum.io/downloads/>`_
to install and/or update these packages.

* `python <http://www.python.org/>`_ versions 2.7, or 3.6 or later
* `numpy <http://www.numpy.org/>`_ version 1.12 or later
* `astropy <http://www.astropy.org/>`_ version 2.0 or later
* `scipy <http://www.scipy.org/>`_ version 0.18 or later
* `tensorflow`_ version 1.4.1 or later
* `prettytensor`_ version 0.7.4 or later  [This dependency may be eventually eliminated]

If you are using Anaconda, you can check the presence of these packages with::

	conda list "^python|numpy|astropy|scipy|matplotlib|pyqt|h5py"

If the packages have been installed, this command should print
out all the packages and their version numbers.

If any of these packages are missing you can install them
with a command like::

	pip install tensorflow

If any of the packages are out of date, they can be updated
with a command like::

	conda update scipy
	# or
	pip update tensorflow


Installing spit
===============

Presently, you must download the code from github::

	#go to the directory where you would like to install specdb.
	git clone https://github.com/PYPIT/spit.git

From there, you can build and install with

	cd spit
	python setup.py install  # or use develop


This should install the package and scripts.
Make sure that your PATH includes the standard
location for Python scripts (e.g. ~/anaconda/bin)



