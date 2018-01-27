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
    parser = argparse.ArgumentParser(description='Run SPIT on an image')
    parser.add_argument("image_file", type=str, help="Image to classify (e.g. r6.fits)")
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
    import warnings

    from auto_type.classify import classify_me

    # Do it
    answer = classify_me(pargs.image_file)

    print("You input the image: {:s}".format(pargs.image_file))
    print("   SPIT classified it as a type:  {:s}".format(answer))
