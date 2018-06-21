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

    if isinstance(pargs, tuple):
        instr = pargs[0]
        niter = pargs[1]
    else:
        instr = pargs.instr
        niter = pargs.niter

    # Instrument specific
    if instr == 'Kast':
        run(instr, num_iterations=niter)
    else:
        raise IOError("Not ready for this instrument")

# Command line execution
if __name__ == '__main__':
    import sys

    instr = sys.argv[1]
    ninter = int(sys.argv[2])

    main((instr, ninter))