from __future__ import print_function, absolute_import, division, unicode_literals

# Requirements:
import numpy as np, glob, os, sys
from astropy.io import fits
from PIL import Image
from enum import Enum

from scipy import misc

from collections import OrderedDict

import pdb

from spit.preprocess import trim_image
from spit.preprocess import zscale
from spit.preprocess import one_hot_encoded
from spit.utils import congrid

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

# Mapping from image type to index
label_dict = OrderedDict()
label_dict['bias_label']=0
label_dict['science_label']=1
label_dict['standard_label']=2
label_dict['arc_label']=3
label_dict['flat_label']=4

def load_linear_pngs(data_type, img_path='/scratch/citrisdance_viktor/'):
    image_data = {}

    # Define the image locations
    if data_type == "train_data":
        load_batch_size = 504
        data_locations = [ \
            img_path+"viktor_astroimage/linear_datasets/bias_train/*", \
            img_path+"viktor_astroimage/linear_datasets/science_train/*", \
            img_path+"viktor_astroimage/linear_datasets/standard_train/*", \
            img_path+"viktor_astroimage/linear_datasets/arc_train/*", \
            img_path+"viktor_astroimage/linear_datasets/flat_train/*", \
            img_path+"viktor_astroimage/linear_datasets/bias_validation/*", \
            img_path+"viktor_astroimage/linear_datasets/science_validation/*", \
            img_path+"viktor_astroimage/linear_datasets/standard_validation/*", \
            img_path+"viktor_astroimage/linear_datasets/arc_validation/*", \
            img_path+"viktor_astroimage/linear_datasets/flat_validation/*", \
            img_path+"viktor_astroimage/linear_datasets/bias_test/*", \
            img_path+"viktor_astroimage/linear_datasets/science_test/*", \
            img_path+"viktor_astroimage/linear_datasets/science_enforced/*", \
            img_path+"viktor_astroimage/linear_datasets/standard_test/*", \
            img_path+"viktor_astroimage/linear_datasets/arc_test/*", \
            img_path+"viktor_astroimage/linear_datasets/flat_test/*"]


    elif data_type == "kast_test_data":
        print("Loading the test data..")
        load_batch_size = 160
        data_locations = [ \
            img_path+"viktor_astroimage/linear_datasets/real_bias_test/*", \
            img_path+"viktor_astroimage/linear_datasets/real_science_test/*", \
            img_path+"viktor_astroimage/linear_datasets/real_standard_test/*", \
            img_path+"viktor_astroimage/linear_datasets/real_arc_test/*", \
            img_path+"viktor_astroimage/linear_datasets/real_flat_test/*"]

    # Construct the dict arrays
    raw_data = []
    labels = []
    filenames = []

    for index, location in enumerate(data_locations):
        images = glob.glob(location)
        image_array = []
        image_labels = []
        image_filenames = []
        for image_file in images:
            image_data = misc.imread(image_file, mode='L')
            padded_image = image_data.flatten()
            image_array.append(padded_image)
            image_label = 0
            if "bias" in image_file:
                image_label = label_dict['bias_label']
            elif "science" in image_file:
                image_label = label_dict['science_label']
            elif "standard" in image_file:
                image_label = label_dict['standard_label']
            elif "arc" in image_file:
                image_label = label_dict['arc_label']
            elif "flat" in image_file:
                image_label = label_dict['flat_label']

            image_labels.append(image_label)
            image_filenames.append(image_file)

        raw_data = raw_data + image_array
        labels = labels + image_labels
        filenames = filenames + image_filenames

    print("Loaded!")
    # Get the raw images.
    raw_images = np.array(raw_data)

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(labels)

    return raw_images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes), filenames

def load_images():
    data_locations = [ \
        "images/bias/linear*", \
        "images/science/linear*", \
        "images/standard/linear*", \
        "images/arc/linear*", \
        "images/flat/linear*"]

    raw_data = []
    labels = []
    filenames = []

    for index, location in enumerate(data_locations):
        images = glob.glob(location)
        image_array = []
        image_labels = []
        image_filenames = []
        for image_file in images:
            image_data = misc.imread(image_file, mode='L')
            padded_image = image_data.flatten()

            image_array.append(padded_image)
            image_labels.append(index)
            image_filenames.append(image_file)

        raw_data = raw_data + image_array
        labels = labels + image_labels
        filenames = filenames + image_filenames

    # Get the raw images.
    raw_images = np.array(raw_data)

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(labels)

    return raw_images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes), filenames

def load_all_data():
    data_locations = [ \
        "images/bias/linear*", \
        "images/science/linear*", \
        "images/standard/linear*", \
        "images/arc/linear*", \
        "images/flat/linear*"]

    raw_data = []
    labels = []
    filenames = []

    for index, location in enumerate(data_locations):
        images = glob.glob(location)
        image_array = []
        image_labels = []
        image_filenames = []
        for image_file in images:
            image_data = misc.imread(image_file, mode='L')
            padded_image = image_data.flatten()

            image_array.append(padded_image)
            image_labels.append(index)
            image_filenames.append(image_file)

        raw_data = raw_data + image_array
        labels = labels + image_labels
        filenames = filenames + image_filenames

    data_locations = [ \
        "/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/bias_train_histequ/*", \
        "/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/science_train_histequ/*", \
        "/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/standard_train_histequ/*", \
        "/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/arc_train_histequ/*", \
        "/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/flat_train_histequ/*"]

    for index, location in enumerate(data_locations):
        images = glob.glob(location)
        image_array = []
        image_labels = []
        image_filenames = []
        for image_file in images:
            image_data = misc.imread(image_file, mode='L')
            padded_image = image_data.flatten()

            image_array.append(padded_image)
            image_labels.append(index)
            image_filenames.append(image_file)

        raw_data = raw_data + image_array
        labels = labels + image_labels
        filenames = filenames + image_filenames

    data_locations = [ \
        "/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/bias_test_histequ/*", \
        "/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/science_test_histequ/*", \
        "/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/standard_test_histequ/*", \
        "/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/arc_test_histequ/*", \
        "/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/flat_test_histequ/*"]

    for index, location in enumerate(data_locations):
        images = glob.glob(location)
        image_array = []
        image_filenames = []
        for image_file in images:
            image_data = misc.imread(image_file, mode='L')
            padded_image = image_data.flatten()

            image_array.append(padded_image)
            image_labels.append(index)
            image_filenames.append(image_file)

        raw_data = raw_data + image_array
        labels = labels + image_labels
        filenames = filenames + image_filenames
    # Get the raw images.
    raw_images = np.array(raw_data)

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(labels)

    return raw_images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes), filenames


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

