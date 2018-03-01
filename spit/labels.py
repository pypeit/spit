""" SPIT Images object
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import os
from pkg_resources import resource_filename

from collections import OrderedDict

# KAST
def kast_label_dict():
    # Mapping from image type to index
    label_dict = OrderedDict()
    label_dict['bias_label']=0
    label_dict['science_label']=1
    label_dict['standard_label']=2
    label_dict['arc_label']=3
    label_dict['flat_label']=4
    # Return
    return label_dict


def kast_classify_dict(label_dict=None):
    if label_dict is None:
        label_dict = kast_label_dict()
    # Classify dict
    classify_dict = {}
    for key,item in label_dict.items():
        classify_dict[item] = key.split('_')[0]
    classify_dict[-1] = 'unknown'
    # Return
    return classify_dict

