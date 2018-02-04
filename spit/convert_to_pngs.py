"""
Convert FITS files to PNGs
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np, glob
import scipy
from astropy.io import fits
from astropy.stats import sigma_clip

from auto_type.image_loader import cutoff_back, cutoff_forw


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