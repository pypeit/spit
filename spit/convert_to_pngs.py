"""
Convert FITS files to PNGs
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np, glob
import scipy
from astropy.io import fits
from astropy.stats import sigma_clip

from auto_type.image_loader import cutoff_back, cutoff_forw

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

def convert_images(data_locations):
    """
    Parameters
    ----------
    data_locations : list
      Folders to use for image generation

    Returns
    -------

    """
    for index, location in enumerate(data_locations):
        images = glob.glob(location)
        for image_file in images:
            image_title = image_file.split("/")
            names = image_title[-1].split(".")
            cls = image_title[1]
            filename = names[0]

            fits_f = fits.open(image_file, 'readonly')
            hdu = fits_f[0]
            image = hdu.data
            convert_image(image)


def convert_image(image):
    shape = image.shape

    # Rotate?
    if shape[0] > shape[1]:
        image = scipy.ndimage.interpolation.rotate(image, angle=90.0)
        filtered_data = sigma_clip(image, sigma=3, axis=0).tolist()

        max_vals = []
        for idx, val in enumerate(filtered_data):
            max_vals.append(max(val))
        cutoff_f = cutoff_forw(max_vals)
        cutoff_b = cutoff_back(max_vals)
        first = cutoff_f
        second = shape[0] - cutoff_b

        image = image[first:second, :]

        v1.load_data(image)
    else:
        v1.load_data(image)

    filename = "./images/" + cls + "/linear_" + filename + ".png"
    v1.save_rgb_image_as_file(filename, format='png')
    print(filename)