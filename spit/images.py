""" SPIT Images object
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import os
from pkg_resources import resource_filename
import tensorflow as tf
import prettytensor as pt

from spit.image_loader import load_linear_pngs

class Images(object):
    """ Class to hold training, test, etc. images
    And their labels

    Attributes
    ----------
    images :  np.array
    cls : np.array
    labels : list
    filenames : list
    """

    def __init__(self, img_type, **kwargs):
        """
        Parameters
        ----------
        img_type : str
          Type of images loaded
            kast_test_data : Validation dataset for original Kast work (Yankoff & Prochaska 2018)
        kwargs
        """
        # Check
        if img_type not in['kast_test_data']:
            raise IOError('Not ready for img_type: {:s}'.format(img_type))

        # Load
        self.images, self.cls, self.labels, self.filenames = load_linear_pngs(data_type=img_type)


