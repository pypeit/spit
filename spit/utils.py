""" Module for base-level methods for SPIT"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import scipy.interpolate
import scipy.ndimage
import pdb
import matplotlib.pyplot as plt

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print("[congrid] dimensions error. " \
        "This routine currently only support " \
        "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range(ndims):
            base = np.indices(newdims)[i]
            dimlist.append((old[i] - m1) / (newdims[i] - m1) \
                           * (base + ofs) - ofs)
        cd = np.array(dimlist).round().astype(int)
        newa = a[list(cd)]
        return newa

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append((old[i] - m1) / (newdims[i] - m1) \
                           * (base + ofs) - ofs)
        # specify old dims
        olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d(olddims[-1], a, kind=method)
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + list(range(ndims - 1))
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)
            try:
                mint = scipy.interpolate.interp1d(olddims[i], newa, kind=method, fill_value='extrapolate')
            except ValueError:
                pdb.set_trace()

            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

        return newa
    elif method in ['spline']:
        oslices = [slice(0, j) for j in old]
        oldcoords = np.ogrid[oslices]
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        # make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n", \
        "Currently only \'neighbour\', \'nearest\',\'linear\',", \
        "and \'spline\' are supported.")
        return None


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

def display_training_trends(history, key1, key2, title='training'):
    """
    Display the graphs of loss/accuracy during training
    
    :param history:
        Tensorflow History object or Numpy Decompressed File object.
    :param key1:
        Key contained within history object that user can provide to plot as compared with other keys.
        Assume key is a string contained in the history object.
    :param key2:
        Key contained within history object that user can provide to plot as compared with other keys.
        Assume key is a string contained in the history object.
    :param title:
        Title of the graph. If caller doesn't specify, use default of 'training'.
    """
    h = None
    if  'tensorflow' in str(type(history)):
      h = history.history
    else:
      h = history
    # make subplots
    fig, ax1 = plt.subplots()

    # set title of graph
    plt.title(title) 

    # set x label
    color = 'tab:red'
    ax1.set_xlabel('epochs')

    # plot first key
    ax1.set_ylabel(key1, color=color)  # we already handled the x-label with ax1
    lns1 = ax1.plot(h[key1], color=color, label = key1) # maybe improve labeling 
    ax1.tick_params(axis='y')

    color = 'tab:blue'
    legend_loc = 'best'

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  

    # plot second key
    ax2.set_ylabel(key2, color=color)  # we already handled the x-label with ax1
    lns2 = ax2.plot(h[key2], color=color, label = key2)
    ax2.tick_params(axis='y') #labelcolor=color

    # legend for two axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=legend_loc)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
