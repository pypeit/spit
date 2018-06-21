# I/O for SPIT
from __future__ import print_function, absolute_import, division, unicode_literals

import pdb


def write_array_to_png(img, outfile, verbose=False):
    """ Write input array to PNG
    Parameters
    ----------
    img : ndarray
    outfile : str
    verbose : bool, optional

    Returns
    -------

    """
    from PIL import Image
    pil_image = Image.fromarray(img)
    pil_image.save(outfile, format='png')
    if verbose:
        print("Wrote image to {:s}".format(outfile))


def read_fits(image_file, exten=0):
    """

    Parameters
    ----------
    image_file
    exten : int, optional
      Extension the FITS HDUList

    Returns
    -------
    image : ndarray

    """
    from astropy.io import fits

    # Open the fits file
    fits_f = fits.open(image_file, 'readonly')
    hdu = fits_f[exten]
    image = hdu.data

    # Return
    return image


def write_classifier_predictions(classifier, outfile):
    import json

    def native(value):
        converted_value = getattr(value, "tolist", lambda x=value: x)()
        return converted_value

    pdict = {}
    pdict['predictions'] = native(classifier.cls_pred)

    with open(outfile, 'wt') as fh:
        json.dump(pdict, fh)
