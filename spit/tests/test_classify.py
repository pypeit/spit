# Module to run tests on generating AbsSystem
from __future__ import print_function, absolute_import, division, unicode_literals

# TEST_UNICODE_LITERALS

from pkg_resources import resource_filename
import os
import warnings

import pytest

from spit.classify import classify_me
from spit.classifier import Classifier

def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), 'files')
    return os.path.join(data_dir, filename)


def test_classify_arc():
    save_dir = resource_filename('spit', '/data/checkpoints/kast_original/')
    # Classifier
    kast = Classifier.load_kast()
    if os.path.isdir(save_dir):
        # Tests from_dict too
        _, _, answer = classify_me(data_path('r6.fits'), kast)
        assert answer.upper() == 'ARC'



