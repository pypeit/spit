#!/usr/bin/env python
"""
Train a SPIT classifier
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
    parser = argparse.ArgumentParser(description='Train a SPIT classifier [v1]')
    parser.add_argument("instr", type=str, help="Instrument to train [Kast]")
    parser.add_argument("niter", type=int, help="Number of training iterations")

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

    from spit.training import run

    # Instrument specific
    if pargs.instr == 'Kast':
        run(pargs.instr, num_iterations=pargs.niter)
    else:
        raise IOError("Not ready for this instrument")

