""" Module for the figures of FRB Halos paper
"""

# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, os, sys, json
import pdb
from pkg_resources import resource_filename

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


from astropy.io import fits

from spit import preprocess as spit_p
from spit import classify as spit_c
from spit.classifier import Classifier


# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import coshalo_lls as chlls


def setup_image_set(set=['all']):
    # Images
    img_path = os.getenv('HOME')+'/Lick/Kast/data/2014aug28/Raw/'
    images = []
    images.append(dict(type='Bias', frame='b227'))
    images.append(dict(type='Flat', frame='b295'))
    images.append(dict(type='Arc', frame='b294'))
    images.append(dict(type='Standard', frame='b289'))
    images.append(dict(type='Science', frame='b264'))
    # Parse
    if set[0] == 'all':
        final_images = images
    else:
        final_images = []
        for kk,image in enumerate(images):
            if image['type'] in set:
                final_images.append(images[kk].copy())
    # Return
    return img_path, final_images

def fig_images(field=None, outfil=None):
    """ Spectral images
    """
    # Init
    if outfil is None:
        outfil = 'fig_spec_images.png'
    img_path, images = setup_image_set()

    # Targets only
    plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(len(images),1)

    #plt.suptitle('{:s}: MMT/Hectospec Targets'.format(field[0])
    #    ,fontsize=19.)

    # Loop me
    cm = plt.get_cmap('Greys')
    for tt,image in enumerate(images):
        ax = plt.subplot(gs[tt])

        # Load
        hdu = fits.open(img_path+image['frame']+'.fits.gz')
        img = hdu[0].data

        # z-Scale
        z1,z2 = spit_p.zscale(img, only_range=True)
        #pdb.set_trace()


        # Plot
        ax.imshow(img, vmin=z1, vmax=z2, cmap=cm)#, extent=(imsize/2., -imsize/2, -imsize/2.,imsize/2))

        # Axes
        ax.axis('off')

        # Labels
        ax.text(0.07, 0.90, image['type'], transform=ax.transAxes, color='b',
                size='large', ha='left', va='top')
        #ax_hecto.set_xlim(imsize/2., -imsize/2.)
        #ax_hecto.set_ylim(-imsize/2., imsize/2.)

    plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.0)
    plt.savefig(outfil, dpi=700)
    plt.close()
    print("Wrote: {:s}".format(outfil))

def fig_zscale(field=None, outfil=None):
    """ Compare two views of the same image.
    With and without ZSCALE
    """
    # Init
    if outfil is None:
        outfil = 'fig_zscale.png'
    img_path, images = setup_image_set(set=['Bias'])
    # Load bias
    hdu = fits.open(img_path+images[0]['frame']+'.fits.gz')
    img = hdu[0].data

    # Targets only
    plt.figure(figsize=(12, 5))
    plt.clf()
    gs = gridspec.GridSpec(2,1)

    #plt.suptitle('{:s}: MMT/Hectospec Targets'.format(field[0])
    #    ,fontsize=19.)

    cm = plt.get_cmap('Greys')

    # Without zscale
    ax0 = plt.subplot(gs[0])
    # Plot
    ax0.imshow(img, cmap=cm, vmin=0, vmax=1062)
    # Axes
    ax0.axis('off')


    # With zscale
    ax1 = plt.subplot(gs[1])
    # z-Scale
    z1,z2 = spit_p.zscale(img, only_range=True)

    # Plot
    ax1.imshow(img, vmin=z1, vmax=z2, cmap=cm)#, extent=(imsize/2., -imsize/2, -imsize/2.,imsize/2))
    # Axes
    ax1.axis('off')

    # Labels
    #ax.text(0.07, 0.90, image['type'], transform=ax.transAxes, color='b',
    #        size='large', ha='left', va='top')

    plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.0)
    plt.savefig(outfil, dpi=700)
    plt.close()
    print("Wrote: {:s}".format(outfil))


def fig_find_trimsec(outfile=None):
    """ DEIMOS completeness figure
        Using the MAG_MAX in the YAML files
    """
    if outfile is None:
        outfile = 'fig_find_trimsec.pdf'

    # Load image
    arc_file = resource_filename('auto_type', 'tests/files/r6.fits')
    hdulist = fits.open(arc_file)
    img = hdulist[0].data

    #
    timg, stuff = spit_p.trim_image(img, ret_all=True)
    max_vals, cutoff_f, cutoff_b = stuff


    # Plot
    plt.figure(figsize=(5, 4))
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    ax.plot(max_vals, 'k', drawstyle='steps-mid')
    # Lines
    ax.axvline(cutoff_f, color='b', ls='dashed')
    ax.axvline(len(max_vals)-cutoff_b, color='b', ls='dashed')


    # Labels
    ax.set_xlabel('Row')
    ax.set_ylabel('Maximum Value')
    #ax.set_xlim(17., 24.)
    #ax.set_ylim(0., 1.05)

    set_fontsize(ax, 13.)
    # Legend
    #legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.2,
    #                    handletextpad=0.1, fontsize='large')

    plt.tight_layout(pad=0.2,h_pad=0.3,w_pad=0.0)
    plt.savefig(outfile, dpi=700)
    plt.close()
    print("Wrote: {:s}".format(outfile))


def fig_trim(field=None, outfil=None):
    """ Compare two views of the same image.
    With and without ZSCALE
    """
    # Init
    if outfil is None:
        outfil = 'fig_trim.png'

    # Load image
    arc_file = resource_filename('auto_type', 'tests/files/r6.fits')
    hdulist = fits.open(arc_file)
    img = hdulist[0].data

    # Targets only
    plt.figure(figsize=(12, 5))
    plt.clf()
    gs = gridspec.GridSpec(2,1)

    #plt.suptitle('{:s}: MMT/Hectospec Targets'.format(field[0])
    #    ,fontsize=19.)

    cm = plt.get_cmap('Greys')

    # Untrimmed
    z1,z2 = spit_p.zscale(img, only_range=True)

    ax0 = plt.subplot(gs[0])
    # Plot
    ax0.imshow(img, cmap=cm, vmin=z1, vmax=z2)
    # Axes
    ax0.axis('off')


    # Trimmed
    timg = spit_p.trim_image(img)
    ax1 = plt.subplot(gs[1])
    # z-Scale
    z1,z2 = spit_p.zscale(timg, only_range=True)

    # Plot
    ax1.imshow(timg, vmin=z1, vmax=z2, cmap=cm)#, extent=(imsize/2., -imsize/2, -imsize/2.,imsize/2))
    # Axes
    ax1.axis('off')

    # Labels
    #ax.text(0.07, 0.90, image['type'], transform=ax.transAxes, color='b',
    #        size='large', ha='left', va='top')

    plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.0)
    plt.savefig(outfil, dpi=700)
    plt.close()
    print("Wrote: {:s}".format(outfil))


def fig_test_accuracy(outfile=None, cm=None, return_cm=False):
    """ Test accuracy figure
    """
    from spit.train import print_test_accuracy
    from spit.images import Images
    from sklearn.metrics import confusion_matrix
    from spit import preprocess
    from spit.image_loader import label_dict
    num_classes = preprocess.num_classes

    # Init
    if outfile is None:
        outfile = 'fig_test_accuracy.png'

    if cm is None:
        # Load classifier and initialize
        classifier = Classifier(resource_filename('spit', '/data/checkpoints/kast_original/best_validation'))

        # Load images
        images = Images('kast_test_data')

        # Run me
        print_test_accuracy(classifier, images,
                            show_confusion_matrix=False, show_example_errors=False)

        cls_true = images.cls

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_true,
                              y_pred=classifier.cls_pred)
        if return_cm:
            return cm

    # Print the confusion matrix as text.
    print(cm)

    plt.figure(figsize=(12, 5))
    plt.clf()

    # Plot the confusion matrix as an image.
    mpl = plt.matshow(cm)

    # Make various adjustments to the plot.
    cb = plt.colorbar(mpl, fraction=0.030, pad=0.04)
    cb.set_label('N Images')

    # Labels
    labels = []
    for key in label_dict.keys():
        labels.append(key.split('_'))

    '''
    tick_marks = np.array(labels)
    plt.xticks(range(len(labels)), tick_marks)
    plt.yticks(range(len(labels)), tick_marks)
    '''
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Finish
    #plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.0)
    plt.savefig(outfile, dpi=700)
    plt.close()
    print("Wrote: {:s}".format(outfile))

    return cm

def set_fontsize(ax,fsz):
    '''
    Generate a Table of columns and so on
    Restrict to those systems where flg_clm > 0

    Parameters
    ----------
    ax : Matplotlib ax class
    fsz : float
      Font size
    '''
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsz)


#### ########################## #########################
def main(flg_fig):

    if flg_fig == 'all':
        flg_fig = np.sum( np.array( [2**ii for ii in range(5)] ))
    else:
        flg_fig = int(flg_fig)

    # Spectral images
    if flg_fig & (2**0):
        fig_images()

    # zscale
    if flg_fig & (2**1):
        fig_zscale()

    # Trim
    if flg_fig & (2**2):
        fig_find_trimsec()#'fig_find_trimsec.png')
        fig_trim()

    # Test Accuracy
    if flg_fig & (2**3):
        fig_test_accuracy()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2**0   # Spectral images
        #flg_fig += 2**1   # zscale
        #flg_fig += 2**2   # trim
        flg_fig += 2**3   # Test accuracy
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
