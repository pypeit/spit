.. highlight:: rest

************
Running SPIT
************

This file offers a few notes on running
SPIT within a Python code.  You might
find this `iPython Notebook <https://github.com/PYPIT/spit/blob/master/docs/nb/Running_SPIT.ipynb>`_
to be the most helpful.

Load the Classifier
===================

SPIT needs a CNN arachitecture to run.  Here is
the call to load the Kast classifier::

    from spit.classifier import Classifier
    kast = Classifier.load_kast()

Pre-Process
===========

Assuming you have an image in memory (as a Numpy ndarray),
presume to pre-process::

    pdict = spit_p.original_preproc_dict()  # dict to guide the steps
    images_array = spit_p.flattened_array(data, pdict) # Generates 4 flattened images

Classify
========

Time to classify.  Here goes::

    prediction, results = spit_c.get_prediction(images_array, kast)

And this will turn the output into an image_type name::

    pred_type = kast.classify_dict[prediction]

And that is that.

