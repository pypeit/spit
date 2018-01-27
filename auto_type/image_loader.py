from __future__ import print_function, absolute_import, division, unicode_literals

# Requirements:
import numpy as np, glob, os, sys
from astropy.io import fits
from PIL import Image
from enum import Enum

import pdb

from auto_type.preprocess import trim_image
from auto_type.preprocess import zscale
from auto_type.utils import congrid

sys.dont_write_bytecode = True

class Frames(Enum):
    BIAS        = 0
    SCIENCE     = 1
    STANDARD    = 2
    ARC         = 3
    FLAT        = 4

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
# The height of an image
image_height = 210

# The width of an image
image_width = 650

# Length of an image when flattened to a 1-dim array.
img_size_flat = image_height * image_width

# Tuple with height and width of images used to reshape arrays.
img_shape = (image_height, image_width)

# Number of channels in each image, 1 channel since it's a Gray image
num_channels = 1

# Number of classes.
num_classes = 5

# The overscan cutoff percent difference. If the running average is different
# from the previous running average by this much, then detect an overscan region
cutoff_percent = 1.10

'''
def get_viewer_and_server():
    # Set this to True if you have a non-buggy python OpenCv bindings--it greatly speeds up some operations
    use_opencv = False

    server = ipg.make_server(host='localhost', port=9915, use_opencv=use_opencv)

    # Start viewer server
    # IMPORTANT: if running in an IPython/Jupyter notebook, use the no_ioloop=True option
    server.start(no_ioloop=True)

    # Get a viewer
    # This will get a handle to the viewer v1 = server.get_viewer('v1')
    v1 = server.get_viewer('v1')

    return v1, server

def get_viewer_address():
    return v1.url

def set_viewer_prefs(v1):
    # set a color map on the viewer 
    v1.set_color_map('gray')

    # Set color distribution algorithm
    # choices: linear, log, power, sqrt, squared, asinh, sinh, histeq, 
    v1.set_color_algorithm('linear')
    # Set cut level algorithm to use
    v1.set_autocut_params('zscale', contrast=0.25)
    # Auto cut levels on the image
    v1.auto_levels()
    # set the window size
    v1.set_window_size(650, 210)

def stop_server(server):
    server.stop()
'''


def load_images_arr(image_file, outfile=None):
    """ Convert an input FITS file into 4 flattened arrays
    having flipped it around

    Parameters
    ----------
    image_file : str
      Name of the FITS file

    Returns
    -------
    images_array : ndarray
      nimgs (4) x flattened image
    """

    #v1, server = get_viewer_and_server()
    #set_viewer_prefs(v1)

    # Paths
    basename = os.path.basename(image_file)
    names = basename.split(".")
    file_root = names[0]

    # Open the fits file
    fits_f = fits.open(image_file, 'readonly')
    hdu = fits_f[0]
    image = hdu.data
    shape = image.shape

    # Trim
    image = trim_image(image, cutoff_percent=cutoff_percent)

    # Resize
    rimage = congrid(image.astype(float), (image_height, image_width))

    # zscale
    zimage = zscale(rimage)

    # Load into PIL
    pil_image = Image.fromarray(zimage)
    if outfile is not None:
        pil_image.save(outfile)

    # Generate flipped images 
    pil_image.transpose(Image.FLIP_TOP_BOTTOM)
    ver_image = np.array(pil_image.transpose(Image.FLIP_TOP_BOTTOM))
    hor_image = np.array(pil_image.transpose(Image.FLIP_LEFT_RIGHT))
    hor_ver_image = np.array(pil_image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM))

    # Add flipped images to an array
    im_array = []
    im_array.append(zimage.flatten())
    im_array.append(ver_image.flatten())
    im_array.append(hor_image.flatten())
    im_array.append(hor_ver_image.flatten())
    images_array = np.array(im_array)

    return images_array

