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

def generate_training_pngs(clobber=False):
    outroot = spit_path+'/Kast/PNG/train/'

    # Flats first (they are the most common)
    flat_files = glob.glob(spit_path+'/Kast/FITS/train/flat/0_*fits.gz')
    nflats = len(flat_files)
    # Loop me
    outdir = outroot+'flat/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for flat_file in flat_files:
        # Out_pref
        basename = os.path.basename(flat_file)
        out_pref = basename[2:-8]
        outfiles = glob.glob(outdir+out_pref+'*.png')
        if (len(outfiles) == 4) & (not clobber):
            print("FITS file {:s} already processed".format(basename))
            continue
        else:
            print("Processing FITS file {:s}".format(basename))
        # Load
        data = spit_io.read_fits(flat_file)
        # Process
        image = spit_p.process_image(data, debug=debug)
        # Flip around
        flip_images = spit_p.flips(image, flatten=False)
        # Write PNGs
        for img, suff in zip(flip_images, ['norm','vert','hor','horvert']):
            outfile = outdir+out_pref+'_'+suff+'.png'
            #
            spit_io.write_array_to_png(img, outfile)

    # Other image types (normalizing to the number of flats)
    for itype in ['arc','bias','standard','science']:
        files = glob.glob(spit_path+'/Kast/FITS/train/{:s}/0_*fits.gz'.format(itype))
        nfiles = len(files)
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
