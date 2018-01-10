import numpy as np, glob, math
from astropy.io import fits
from os import walk
import pickle
import json
from scipy import misc
import tensorflow as tf

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

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 1

# Number of classes.
num_classes = 5

# The padding value for the padded image
pad_const = 0
########################################################################


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(cls)-1.
    :return:
        2-dim array of shape: [len(cls), num_classes]
    """

    # Find the number of classes if None is provided.
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1

    return np.eye(num_classes, dtype=float)[class_numbers]

def resize_image(image_data, sess):
    # Convert the image data to an int
    image_data = image_data.astype(int)
    height = image_data.shape[0]
    width = image_data.shape[1]

    # Claculate the aspect ratio
    aspect_ratio = float(width / height)

    # Add a third dimension, this is required by TensorFlow
    image_data = image_data[..., np.newaxis]

    # Define the new sizes keeping the aspect ration
    new_height = int(image_width / aspect_ratio)

    # User tensorflow to resize the image
    new_image = tf.image.resize_images(image_data, [new_height, image_width])
    new_image = new_image.eval(session=sess)
    
    # Remove the third added dimension
    new_image = new_image[:, :, 0]

    # Add padding to make the image 200x600
    height_pad = image_height - new_image.shape[0]
    width_pad = image_width - new_image.shape[1]
    npad = ((0, height_pad), (0, width_pad))
    
    padded_image = np.pad(new_image, pad_width=npad, mode='constant', constant_values=pad_const) 
    
    return padded_image

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
                img_path+"viktor_astroimage/bias_train/*", \
                img_path+"viktor_astroimage/science_train/*", \
                img_path+"viktor_astroimage/standard_train/*", \
                img_path+"viktor_astroimage/arc_train/*", \
                img_path+"viktor_astroimage/flat_train/*"]
        
    elif data_type == "test_data":
        load_batch_size = 1650

        data_locations = [ \
                img_path+"viktor_astroimage/bias_test/*", \
                img_path+"viktor_astroimage/science_test/*", \
                img_path+"viktor_astroimage/standard_test/*", \
                img_path+"viktor_astroimage/arc_test/*", \
                img_path+"viktor_astroimage/flat_test/*"]

    elif data_type == "validation_data":
        load_batch_size = 1350
        data_locations = [ \
                img_path+"viktor_astroimage/bias_validation/*", \
                img_path+"viktor_astroimage/science_validation/*", \
                img_path+"viktor_astroimage/standard_validation/*", \
                img_path+"viktor_astroimage/arc_validation/*", \
                img_path+"viktor_astroimage/flat_validation/*"]
        
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

def load_linear_pngs(data_type, img_path='/soe/vjankov/scratchdisk/'):
    image_data = {}
    bias_label = 0
    science_label = 1
    standard_label = 2
    arc_label = 3
    flat_label = 4

    # Define the image locations
    if data_type == "train_data":
        load_batch_size = 504
        data_locations = [ \
                img_path+"viktor_astroimage/linear_datasets/bias_train/*", \
                img_path+"viktor_astroimage/linear_datasets/science_train/*", \
                img_path+"viktor_astroimage/linear_datasets/standard_train/*", \
                img_path+"viktor_astroimage/linear_datasets/arc_train/*", \
                img_path+"viktor_astroimage/linear_datasets/flat_train/*",\
                img_path+"viktor_astroimage/linear_datasets/bias_validation/*", \
                img_path+"viktor_astroimage/linear_datasets/science_validation/*", \
                img_path+"viktor_astroimage/linear_datasets/standard_validation/*", \
                img_path+"viktor_astroimage/linear_datasets/arc_validation/*", \
                img_path+"viktor_astroimage/linear_datasets/flat_validation/*",\
                img_path+"viktor_astroimage/linear_datasets/bias_test/*", \
                img_path+"viktor_astroimage/linear_datasets/science_test/*", \
                img_path+"viktor_astroimage/linear_datasets/science_enforced/*", \
                img_path+"viktor_astroimage/linear_datasets/standard_test/*", \
                img_path+"viktor_astroimage/linear_datasets/arc_test/*", \
                img_path+"viktor_astroimage/linear_datasets/flat_test/*"]

        
    elif data_type == "test_data":
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
		image_label = bias_label
	    elif "science" in image_file:
		image_label = science_label
	    elif "standard" in image_file:
		image_label = standard_label
	    elif "arc" in image_file:
		image_label = arc_label
	    elif "flat" in image_file:
		image_label = flat_label

            image_labels.append(image_label)
            image_filenames.append(image_file)

        raw_data = raw_data + image_array
        labels = labels + image_labels
        filenames = filenames + image_filenames
        
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
	"/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/bias_validation_histequ/*", \
	"/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/science_validation_histequ/*", \
	"/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/standard_validation_histequ/*", \
	"/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/arc_validation_histequ/*", \
	"/soe/vjankov/scratchdisk/viktor_astroimage/histequ_datasets/flat_validation_histequ/*"]
        
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
