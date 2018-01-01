""" Module for the figures of FRB Halos paper
"""

# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, os, sys, json
import warnings
import pdb

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits

from ginga.util import zscale

# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import coshalo_lls as chlls



def fig_images(field=None, outfil=None):
    """ Spectral images
    """
    # Init
    if outfil is None:
        outfil = 'fig_spec_images.png'
    # Images
    img_path = os.getenv('HOME')+'/Lick/Kast/data/2014aug28/Raw/'
    images = []
    images.append(dict(type='Bias', frame='b227'))
    images.append(dict(type='Flat', frame='b295'))
    images.append(dict(type='Arc', frame='b294'))
    images.append(dict(type='Standard', frame='b289'))
    images.append(dict(type='Science', frame='b264'))

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
        z1,z2 = zscale.zscale(img)
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

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        flg_fig += 2**0   # Spectral images
        #flg_fig += 2**1   # Hectospec targeting
        #flg_fig += 2**2   # Hectospec completeness
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
