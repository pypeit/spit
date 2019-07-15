from __future__ import print_function, absolute_import, division, unicode_literals

# Requirements:
import numpy as np, glob, os, sys

#from scipy import misc

import imageio


import pdb

from spit.utils import one_hot_encoded

sys.dont_write_bytecode = True

def load_linear_pngs(instr, data_type, label_dict, debug=False, single_copy=False,
                     spit_path=os.getenv('SPIT_DATA'),
                     subset=None, images_only=False):
    """ Load PNGs

    Parameters
    ----------
    instr : str
        Name of instrument, e.g. 'Kast'
    data_type : str
        Training type, e.g. 'train', 'valid'
    label_dict : dict
      Sets label values
    single_copy : bool, optional
      Only grab one copy (with flips) of each image
    subset : int, optional
        Only grab a subset of the full list, i.e. the number of files provided by this parameter
    images_only : bool, optional
        Return only the images?

    Returns
    ----------
    dset: 
       a Tensorflow Dataset Object containing the images and labels as numpy arrays
    one_hot:
       a Tensorflow one-hot encoded array for the label values
    """

    image_data = {}

    # Define the image locations
    data_locations = []
    for itype in ['flat', 'arc', 'bias','standard','science']:
        if single_copy:
            data_locations.append(spit_path+'/'+instr+'/PNG/{:s}/{:s}/0_*png'.format(data_type, itype))
        else:
            data_locations.append(spit_path+'/'+instr+'/PNG/{:s}/{:s}/*png'.format(data_type, itype))


    '''
    if data_type == "train_data":
        load_batch_size = 504
        data_locations = [ \
            spit_path+"viktor_astroimage/linear_datasets/bias_train/*", \
            spit_path+"viktor_astroimage/linear_datasets/science_train/*", \
            spit_path+"viktor_astroimage/linear_datasets/standard_train/*", \
            spit_path+"viktor_astroimage/linear_datasets/arc_train/*", \
            spit_path+"viktor_astroimage/linear_datasets/flat_train/*", \
            spit_path+"viktor_astroimage/linear_datasets/bias_validation/*", \
            spit_path+"viktor_astroimage/linear_datasets/science_validation/*", \
            spit_path+"viktor_astroimage/linear_datasets/standard_validation/*", \
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
    '''

    # Construct the dict arrays
    raw_data = []
    labels = []
    filenames = []

    for index, location in enumerate(data_locations):
        images = glob.glob(location)
        images.sort() # So that the ordering is the same each time
        image_array = []
        image_labels = []
        image_filenames = []
        for kk, image_file in enumerate(images):
            if debug and (kk == 10):
                break
            if subset is not None:
                if kk == subset:
                    break
            # load image
            image_data = imageio.imread(image_file, pilmode='L')
            #padded_image = image_data.flatten()
            #image_array.append(padded_image)
            image_array.append(image_data)
            # get image's type using long logic, could make faster
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
            else:
                pdb.set_trace()

            image_labels.append(image_label)
            image_filenames.append(image_file)

        raw_data = raw_data + image_array
        labels = labels + image_labels
        filenames = filenames + image_filenames

    print("Loaded!")
    # Get the raw images.
    raw_images = np.array(raw_data)
    ishape = list(raw_images.shape)
    raw_images = raw_images.reshape(ishape+[1])
    assert len(raw_images.shape) == 4

    # Get the class-numbers for each image. Convert to numpy-array.
    lbl_array = np.array(labels) # might change cls to cls_nums cuz cls means something different

    if images_only:
        return raw_images, lbl_array

    # cls needs to be one-hot!
    # raise IOError
    dset = tf.data.Dataset.from_tensor_slices((raw_images, lbl_array)) 
    one_hot = tf.one_hot(indices=lbl_array, depth=len(label_dict), dtype=float)

    return dset, one_hot, \
           filenames




'''
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
'''

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

    # Load FITS
    image = spit_io.read_fits(image_file)

    # Pre-process
    zimage = spit_pre.process_image(image)

    # PNG?
    if outfile is not None:
        spit_io.write_array_to_png(zimage, outfile)

    return images_array


def load_data_cropped200x600(data_type, img_path='/soe/vjankov/scratchdisk/'):
    image_data = {}
    """
    a)
    Load the data from the filenames stored in the data_locations 

    b)
    Add 0 padding to each image. The padding is 2112 pixels width and height
    The 2112 is the widest pixel form the dataset

    c)
    Add labels for each image based on the folder they are coming form
    This is the label names and values

    Bias = 0
    Science = 1
    Arc = 2
    Flat = 3

    d)
    Construct a dictionary with the following properties:
        raw_data = the flattened 2112 np array with raw pixel values
        labels = the corresponding labels for each data element
        filename = the corresponding filename

    """
    # Define the image locations
    if data_type == "train_data":
        load_batch_size = 5400

        data_locations = [ \
            img_path + "viktor_astroimage/bias_train/*", \
            img_path + "viktor_astroimage/science_train/*", \
            img_path + "viktor_astroimage/standard_train/*", \
            img_path + "viktor_astroimage/arc_train/*", \
            img_path + "viktor_astroimage/flat_train/*"]

    elif data_type == "test_data":
        load_batch_size = 1650

        data_locations = [ \
            img_path + "viktor_astroimage/bias_test/*", \
            img_path + "viktor_astroimage/science_test/*", \
            img_path + "viktor_astroimage/standard_test/*", \
            img_path + "viktor_astroimage/arc_test/*", \
            img_path + "viktor_astroimage/flat_test/*"]

    elif data_type == "validation_data":
        load_batch_size = 1350
        data_locations = [ \
            img_path + "viktor_astroimage/bias_validation/*", \
            img_path + "viktor_astroimage/science_validation/*", \
            img_path + "viktor_astroimage/standard_validation/*", \
            img_path + "viktor_astroimage/arc_validation/*", \
            img_path + "viktor_astroimage/flat_validation/*"]

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
            hdulist = fits.open(image_file)
            image_data = hdulist[0].data
            padded_image = image_data.flatten()

            image_array.append(padded_image)
            image_labels.append(index)
            image_filenames.append(image_file)

        """
        while len(image_array) < load_batch_size:
            image_array = image_array + image_array
            image_labels = image_labels + image_labels
            image_filenames = image_filenames + image_filenames

        raw_data = raw_data + image_array[:load_batch_size]
        labels = labels + image_labels[:load_batch_size]
        filenames = filenames + image_filenames[:load_batch_size]
        """

        raw_data = raw_data + image_array
        labels = labels + image_labels
        filenames = filenames + image_filenames

    # Get the raw images.
    raw_images = np.array(raw_data)

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(labels)

    return raw_images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes), filenames
'''
