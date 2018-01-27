.. highlight:: rest

************
spit Scripts
************

This file summarizes the *spit* scripts
which are executed outside a Python shell,
but use Python code.
These are installed
within your standard Python script path (e.g.
~/anaconda/bin).

Classify an Image
=================

Run the classifier on an input image.  The file
is currently expected to be in FITS image format.
Here is the usage::

    usage: spit_image [-h] image_file

    Run SPIT on an image [v1]

    positional arguments:
      image_file  Image to classify (e.g. r6.fits)

    optional arguments:
      -h, --help  show this help message and exit

Here is a call you can try::

    cd spit/spit/tests/files
    spit_image r6.fits

And here is the output from that call::

    2018-01-27 08:51:02.225800: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
    =======================================================
    You input the image: r6.fits
       SPIT classified it as a type:  ARC

If you wish to suppress the warning message from tensorflow, do::

    export TF_CPP_MIN_LOG_LEVEL=2

