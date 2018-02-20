"""
Generate files and perform training related to Kast
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob
import os
import pdb

from spit import io as spit_io
from spit import preprocess as spit_p

spit_path = os.getenv('SPIT_DATA')

def generate_training_pngs(clobber=False, debug=False):
    outroot = spit_path+'/Kast/PNG/train/'

    # Flats first (they are the most common)
    flat_files = glob.glob(spit_path+'/Kast/FITS/train/flat/0_*fits.gz')
    nflats = len(flat_files)
    # Loop me
    outdir = outroot+'flat/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for flat_file in flat_files:

    # Other image types (normalizing to the number of flats)
    for itype in ['arc','bias','standard','science']:
        files = glob.glob(spit_path+'/Kast/FITS/train/{:s}/0_*fits.gz'.format(itype))
        nfiles = len(files)
        # Start looping
        ntot = 0
        nstep = 0
        while ntot < nflats:
            npull = min(nflats-ntot, nfiles)
            # Randomize
            rand = np.random.random(npull)
            srt = np.argsort(rand)
            pdb.set_trace()
            for kk in srt:
        pdb.set_trace()

#### ########################## #########################
def main(flg):

    # Generate PNGs
    if flg & (2**0):
        generate_training_pngs()

    # Generate PNGs
    if flg & (2**1):
        #make_rr_plots('Hectospec')
        make_rr_plots('DEIMOS')

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2**0   # PNGs
        #flg += 2**1   # Generate RedRock plots
    else:
        flg = sys.argv[1]

    main(flg)
