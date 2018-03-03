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

from sklearn.metrics import confusion_matrix

from astropy.io import fits

from spit import preprocess as spit_p
from spit import classify as spit_c
from spit.classifier import Classifier
from spit import labels as spit_lbl

from linetools import utils as ltu

# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import coshalo_lls as chlls


def setup_image_set(set=['all']):
    # Images
    img_path = os.getenv('SPIT_DATA')+'/Kast/FITS/train/'
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
    arc_file = resource_filename('spit', 'tests/files/r6.fits')
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
    arc_file = resource_filename('spit', 'tests/files/r6.fits')
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


def fig_sngl_test_accuracy(outfile='fig_sngl_test.png', cm=None, return_cm=False):
    """ Test accuracy figure for one copy of each test frame
    Includes heuristics
    """
    # Init
    label_dict = spit_lbl.kast_label_dict()

    # Predictions
    pdict = ltu.loadjson('../Analysis/chk_test_images.json')
    ytrue = np.array(pdict['true'])
    ypred = np.array(pdict['predictions'])

    cm = confusion_matrix(y_true=ytrue,
                          y_pred=ypred)

    # Print the confusion matrix as text.
    print(cm)
    diffcm = cm.copy()

    # Subtract expected images off the diagonal to show the difference
    ndiag = np.sum(cm[0,:])
    for ii in range(cm.shape[0]):
        diffcm[ii,ii] -= np.sum(cm[ii,:])

    vmnx = np.max(np.abs(diffcm.flatten()))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    # Plot the confusion matrix as an image.
    #mpl = ax.matshow(cm)#, vmin=0, vmax=100, cmap='tab20c')#vmin=0, vmax=80)#, cmap='hot')
    mpl = ax.matshow(diffcm, vmin=-1*vmnx, vmax=vmnx, cmap='seismic')#, cmap='hot')

    # Make various adjustments to the plot.
    cb = fig.colorbar(mpl, fraction=0.030, pad=0.04)
    cb.set_label(r'$\Delta N$ Images')

    # Labels
    labels = []
    for key in label_dict.keys():
        labels.append(key.split('_')[0])
    ax.set_xticklabels(['','unknown']+labels+[''])
    ax.set_yticklabels(['','unknown']+labels+[''])

    #tick_marks = np.arange(num_classes)
    #plt.xticks(tick_marks, range(num_classes))
    #plt.yticks(tick_marks, range(num_classes))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # Finish
    #plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.0)
    fig.savefig(outfile, dpi=700)
    plt.close()
    print("Wrote: {:s}".format(outfile))


def fig_full_test_accuracy(outfile=None, cm=None, return_cm=False, acc_file=None):
    """ Test accuracy figure for the full test suite (including replication)
    No heuristics
    """
    from spit.training import print_test_accuracy
    from spit.images import KastImages
    from spit import io as spit_io
    import json

    # Init
    label_dict = spit_lbl.kast_label_dict()
    if outfile is None:
        outfile = 'fig_full_test_accuracy.png'

    if cm is None:
        # Load classifier and initialize
        #classifier = Classifier(resource_filename('spit', '/data/checkpoints/kast_original/best_validation'))
        classifier = Classifier.load_kast()

        print("Loading images..")
        images = KastImages('test')#, single_copy=True)#, debug=True)
        cls_true = images.cls

        # Classify images?
        if acc_file is None:
            # Run me
            print("Classifying..")
            print_test_accuracy(classifier, images,
                                show_confusion_matrix=False, show_example_errors=False)
            print("Done")
            # Write to disk
            spit_io.write_classifier_predictions(classifier, 'f_tst_acc.json')
            ypred = classifier.cls_pred
        else:
            with open(acc_file, 'rt') as fh:
                obj = json.load(fh)
            ypred = np.array(obj['predictions'])


        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_true,
                              y_pred=ypred)
        if return_cm:
            return cm

    # Print the confusion matrix as text.
    print(cm)
    diffcm = cm.copy()

    # Subtract expected images off the diagonal to show the difference
    ndiag = np.sum(cm[0,:])
    for ii in range(cm.shape[0]):
        diffcm[ii,ii] -= ndiag
    vmnx = np.max(np.abs(diffcm))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    # Plot the confusion matrix as an image.
    #mpl = ax.matshow(cm, vmin=0, vmax=100, cmap='tab20c')#vmin=0, vmax=80)#, cmap='hot')
    mpl = ax.matshow(diffcm, vmin=-1*vmnx, vmax=vmnx, cmap='seismic')#, cmap='hot')

    # Make various adjustments to the plot.
    cb = fig.colorbar(mpl, fraction=0.030, pad=0.04)
    cb.set_label(r'$\Delta N$ Images')

    # Labels
    labels = []
    for key in label_dict.keys():
        labels.append(key.split('_')[0])
    ax.set_xticklabels(['']+labels+[''])
    ax.set_yticklabels(['']+labels+[''])

    #tick_marks = np.arange(num_classes)
    #plt.xticks(tick_marks, range(num_classes))
    #plt.yticks(tick_marks, range(num_classes))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # Finish
    #plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.0)
    fig.savefig(outfile, dpi=700)
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
        #fig_full_test_accuracy(acc_file='f_tst_acc_all.json')
        fig_sngl_test_accuracy()


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
