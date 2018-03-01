# Definitions for SPIT
from __future__ import print_function, absolute_import, division, unicode_literals


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

# The overscan cutoff percent difference. If the running average is different
# from the previous running average by this much, then detect an overscan region
cutoff_percent = 1.10

########################################################################


def standard():
    class Frames(Enum):
        UNKNOWN     = -1  # Only for classification
        BIAS        = 0
        SCIENCE     = 1
        STANDARD    = 2
        ARC         = 3
        FLAT        = 4


