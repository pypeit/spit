# Module to run tests on generating AbsSystem
from __future__ import print_function, absolute_import, division, unicode_literals

# TEST_UNICODE_LITERALS

import numpy as np
import os
import warnings

import pytest

from auto_type.classify import classify_me

def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), 'files')
    return os.path.join(data_dir, filename)


def test_classify_arc():
    # Tests from_dict too
    answer = classify_me(data_path('r6.fits'))
    assert answer == 'ARC'
    # Failure
    with pytest.raises(IOError):
        answer = classify_me(data_path('r6.fits'), save_dir='this_better_fail')



