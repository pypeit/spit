"""
Generate files and perform training related to Kast
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob
import os
import pdb

from spit import convert_to_pngs as spit_png

spit_path = os.getenv('SPIT_DATA')

def generate_training_pngs(clobber=False, debug=False):
    outroot = spit_path+'/Kast/PNG/train/'

    # Flats first (they are the most common)
    flat_files = glob.glob(spit_path+'/Kast/FITS/train/flat/0_*fits.gz')
    nflats = len(flat_files)
    # Output dir
    outdir = outroot+'flat/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # Loop me
    for flat_file in flat_files:
        spit_png.make_standard(flat_file, outdir, [2,-8], 0, clobber=clobber)


    # Other image types (normalizing to the number of flats)
    for itype in ['arc','bias','standard','science']:
        files = glob.glob(spit_path+'/Kast/FITS/train/{:s}/0_*fits.gz'.format(itype))
        nfiles = len(files)
        # Output dir
        outdir = outroot+'{:s}/'.format(itype)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Start looping
        ntot = 0  # Number of FTIS files used
        step = 0  # Index looping through the image for normalization
        # Loop me
        while ntot < nflats:
            npull = min(nflats-ntot, nfiles)
            # Randomize
            rand = np.random.random(npull)
            srt = np.argsort(rand)
            # Loop
            for kk in srt:
                filen = files[kk]
                spit_png.make_standard(filen, outdir, [2,-8], step, clobber=clobber)
            # Increment
            step += 1
            ntot += npull

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
