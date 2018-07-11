# Module to run tests on generating AbsSystem
from __future__ import print_function, absolute_import, division, unicode_literals

# TEST_UNICODE_LITERALS

import numpy as np
import os
import warnings

import pytest

from spit.io import read_fits
from spit import preprocess as spit_p

def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), 'files')
    return os.path.join(data_dir, filename)


def test_load_images():
    # Tests from_dict too
    data = read_fits(data_path('r6.fits'))
    # Process dict
    pdict = spit_p.original_preproc_dict()
    # Process
    images_array = spit_p.flattened_array(data, pdict)
    assert images_array.shape[0] == 4



