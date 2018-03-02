#!/usr/bin/env python
"""
Build PNGs from a FITS file with SPIT pre-processing
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
    parser = argparse.ArgumentParser(description='Build PNGs from a FITS file with SPIT pre-processing [v1]')
    parser.add_argument("image_file", type=str, help="Image to use (e.g. r6.fits)")

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

    from spit.generate_pngs import make_standard
    from spit.preprocess import original_preproc_dict

    pdict = original_preproc_dict()

    # Do it
    i0 = 0
    i1 = pargs.image_file.find('.fits')
    make_standard(pargs.image_file, './', [i0,i1], 0, pdict)

    print("=======================================================")
    print("See the 0_ files in your folder: ")
