""" Module for analyzing aspects of SPIT
"""

# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, os, sys, json
import pdb
from pkg_resources import resource_filename

from astropy.io import fits

from spit import classify as spit_c
from spit.classifier import Classifier
from spit import io as spit_io

from spit import preprocess as spit_p
# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import coshalo_lls as chlls


def chk_test_images():
    # Load SPIT
    classifier = Classifier.load_kast()
    pdict = spit_p.original_preproc_dict()

    # Images
    path = os.getenv('SPIT_DATA')+'/Kast/FITS/test/'
    all_files, true_values, predicted_values = [], [], []

    for itype in ['flat', 'arc', 'bias','standard','science']:
        # Grab the names
        files = glob.glob(path+'/{:s}/0_*fits.gz'.format(itype))
        files.sort()

        # Loop on em
        for ifile in files:
            # Load
            data = spit_io.read_fits(ifile)
            images_array = spit_p.flattened_array(data, pdict)
            # Prediction
            prediction, results = spit_c.get_prediction(images_array, classifier)
            pred_type = classifier.classify_dict[prediction]
            print("{:s}: Input image {:s} is classified as a {:s}".format(itype, os.path.basename(ifile)[:-7],
                                                                pred_type), results)
            # Save
            all_files.append(os.path.basename(ifile))
            predicted_values.append(int(prediction))
            true_values.append(int(classifier.label_dict[itype+'_label']))
            #if itype != pred_type:
            #    pdb.set_trace()
    # Write
    odict = {}
    odict['predictions'] = predicted_values
    odict['true'] = true_values
    odict['files'] = all_files

    with open('chk_test_images.json', 'wt') as fh:
        json.dump(odict, fh)


#### ########################## #########################
def main(flg_anly):

    # Spectral images
    if flg_anly & (2**0):
        chk_test_images()



# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_anly = 0
        flg_anly += 2**0   # Spectral images
    else:
        flg_anly = sys.argv[1]

    main(flg_anly)

