# Module to run tests on generating AbsSystem
from __future__ import print_function, absolute_import, division, unicode_literals

# TEST_UNICODE_LITERALS

import numpy as np
import os
import warnings

import pytest

from spit.image_loader import load_images_arr

def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), 'files')
    return os.path.join(data_dir, filename)


def test_load_images():
    # Tests from_dict too
    img_array = load_images_arr(data_path('r6.fits'), outfile=data_path('tmp.png'))
    assert img_array.shape[0] == 4



