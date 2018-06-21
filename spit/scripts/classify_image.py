#!/usr/bin/env python
"""
Run SPIT on an image
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import pdb

try:  # Python 3
    ustr = unicode
except NameError:
    ustr = str

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Run SPIT on an image [v1]')
    parser.add_argument("image_file", type=str, help="Image to classify (e.g. r6.fits)")
    parser.add_argument("--exten", type=int, default=0, help="Extension (default=0)")
    #parser.add_argument("--zmax", type=float, help="Maximum redshift for analysis")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np

    from spit.classify import classify_me
    from spit.classifier import Classifier

    # Classifier
    kast = Classifier.load_kast()

    # Do it
    _, _, answer = classify_me(pargs.image_file, kast, exten=pargs.exten)

    print("=======================================================")
    print("You input the image: {:s}, extension={:d}".format(pargs.image_file, pargs.exten))
    print("   SPIT classified it as a type:  {:s}".format(answer))
