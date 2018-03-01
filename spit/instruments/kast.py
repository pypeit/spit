"""
Generate files and perform training related to Kast
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob
import os
import pdb

from spit import generate_pngs as spit_png
from spit import preprocess

spit_path = os.getenv('SPIT_DATA')

def generate_pngs(category, clobber=False, seed=12345, debug=False, regular=True):
    """
    Parameters
    ----------
    category : str
    clobber : bool, optional
    debug : bool, optional

    Returns
    -------

    """
    # Pre-processing dict
    pdict = preprocess.original_preproc_dict()

    #
    rstate = np.random.RandomState(seed)
    outroot = spit_path+'/Kast/PNG/{:s}/'.format(category)

    # Flats first (they are the most common)
    flat_files = glob.glob(spit_path+'/Kast/FITS/{:s}/flat/0_*fits.gz'.format(category))
    nflats = len(flat_files)
    # Output dir
    outdir = outroot+'flat/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # Loop me
    for flat_file in flat_files:
        spit_png.make_standard(flat_file, outdir, [2,-8], 0, clobber=clobber)

    # Other image types (regularizing to the number of flats)
    for itype in ['arc','bias','standard','science']:
        files = glob.glob(spit_path+'/Kast/FITS/{:s}/{:s}/0_*fits.gz'.format(category, itype))
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
            # Randomize, but use seeded to avoid new ones appearing!
            rand = rstate.rand(npull)
            srt = np.argsort(rand)
            #if len(np.unique(srt)) != npull:
            #    pdb.set_trace()
            #if npull < nfiles:
            #    pdb.set_trace()
            # Loop
            #save_files = []
            for kk in srt:
                filen = files[kk]
                #if step == 5:
                #    print(kk, filen)
                #save_files.append(filen)
                spit_png.make_standard(filen, outdir, [2,-8], step, pdict, clobber=clobber)
            # Check (Debugging)
            #for ifile in save_files:
            #    if 'may19_2015_r1' in ifile:
            #        print(ifile)
            #if step == 5:
            #    pdb.set_trace()
            # Increment
            step += 1
            ntot += npull

    # Sanity check
    if regular:
        for itype in ['flat', 'arc','bias','standard','science']:
            outroot = spit_path+'/Kast/PNG/{:s}/{:s}'.format(category, itype)
            files = glob.glob(outroot+'/*.png')
            try:
                assert len(files) == 4*nflats
            except AssertionError:
                pdb.set_trace()


#### ########################## #########################
def main(flg):

    # Generate PNGs
    if flg & (2**0):
        generate_pngs('train')
        generate_pngs('test', regular=True)  # Also regularized
        generate_pngs('validation', regular=True)  # Also regularized

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
