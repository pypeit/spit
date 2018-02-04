"""
Generate files and perform training related to Kast
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob
import os
import scipy

spit_path = os.getenv('SPIT_DATA')

def generate_training_pngs():
    # Flats first (they are the most common)
    flat_files = glob.glob(spit_path+'/train/flat/0_*fits')

    # Loop me
    for flat_file in flat_files:
        # Load
