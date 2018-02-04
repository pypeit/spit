# I/O for SPIT
from __future__ import print_function, absolute_import, division, unicode_literals


def write_array_to_png(img, outfile, verbose=False):
    from PIL import Image
    pil_image = Image.fromarray(img)
    pil_image.save(outfile)
    if verbose:
        print("Wrote image to {:s}".format(outfile))


def read_fits(image_file):
    from astropy.io import fits

    # Open the fits file
    fits_f = fits.open(image_file, 'readonly')
    hdu = fits_f[0]
    image = hdu.data

    # Return
    return image

