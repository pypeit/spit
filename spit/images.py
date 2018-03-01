""" SPIT Images object
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import os
from pkg_resources import resource_filename

from spit.image_loader import load_linear_pngs
from spit import labels as spit_lbls
from collections import OrderedDict

class KastImages(object):
    """ Class to hold training, test, etc. images for Kast
    Only 1 set at a time. And their labels

    This may become the child of a more general Image class

    Attributes
    ----------
    images :  np.array
    cls : np.array
    labels : list
    filenames : list
    """

    def __init__(self, iset, **kwargs):
        """
        Parameters
        ----------
        iset : str
          Set of images loaded
            test : Test dataset for original Kast work (Yankoff & Prochaska 2018)
        kwargs
        """
        # Instrument
        self.instr = 'Kast'

        # Check
        if iset not in['train', 'test', 'validation']:
            raise IOError('Not ready for image set: {:s}'.format(iset))
        else:
            print("Loading the set of {:s} images..".format(iset))
        self.set = iset

        # Init defs
        self.init_label_dicts()

        # Load
        self.load_pngs(**kwargs)

    def init_label_dicts(self):
        # Init
        self.label_dict = spit_lbls.kast_label_dict()
        self.classify_dict = spit_lbls.kast_classify_dict(self.label_dict)


    def load_pngs(self, **kwargs):
        # Load
        self.images, self.cls, self.labels, self.filenames = load_linear_pngs(
            self.instr, self.set, self.label_dict, **kwargs)



