""" Module for tables of the Auto Typing paper
"""

# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, os, sys
import warnings
import pdb

from pkg_resources import resource_filename

from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time

from linetools import utils as ltu

from spit import labels as spit_lbl


ddict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 
    'JUN': 6, 'JUNE': 6, 'JUL': 7, 'JULY': 7, 'AUG': 8, 
    'SEP': 9, 'SEPT': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#sys.path.append(os.path.abspath("../Vetting/py"))
#from vette_dr7 import load_ml_dr7


def mktab_images(outfil='tab_images.tex', sub=False):

    # Path
    path = os.getenv('SPIT_DATA')+'/Kast/FITS/'
    if sub:
        outfil = 'tab_images_sub.tex'

    # Scan image sets
    types, dates, frames, pis, flags, sets = [], [], [], [], [], []
    ntest, ntrain, nvalid = 0, 0, 0
    for iset in ['test', 'train', 'validation']:
        for itype in ['arc', 'bias', 'flat', 'science', 'standard']:
            files = glob.glob(path+'{:s}/{:s}/*fits.gz'.format(iset,itype))
            nimg = len(files)
            print("There are {:d} images of type {:s} in set {:s}".format(nimg, itype, iset))
            #
            types += [itype]*nimg
            sets += [iset]*nimg
            if iset == 'test':
                ntest += nimg
            elif iset == 'train':
                ntrain += nimg
            elif iset == 'validation':
                nvalid += nimg
            # Loop on the images
            for ipath in files:
                # Parse
                ifile = os.path.basename(ipath)
                if 'xavier' in ifile:
                    pis += ['Prochaska']
                    # Find date
                    year = ifile[20:24]
                    try:
                        month = ddict[ifile[24:27].upper()]
                    except KeyError:
                        pdb.set_trace()
                    day = ifile[27:29]
                    dates.append('{:s}-{:d}-{:s}'.format(year,month,day))
                    # Frame
                    i1 = ifile.find('.fits')
                    frames.append(ifile[34:i1])
                else:
                    pis += ['Hsyu']
                    # Find date
                    i0 = ifile.find('_20')
                    year = ifile[i0+1:i0+5]
                    try:
                        month = ddict[ifile[2:6].upper()]
                    except:
                        month = ddict[ifile[2:5].upper()]
                        day = ifile[5:i0]
                    else:
                        day = ifile[6:i0]
                    dates.append('{:s}-{:d}-{:s}'.format(year,month,day))
                    # Frame
                    i1 = ifile.find('.fits')
                    frames.append(ifile[i0+6:i1])

    # Summarize
    print('-----------------------------------------')
    print('There are a total of {:d} training images'.format(ntrain))
    print('There are a total of {:d} test images'.format(ntest))
    print('There are a total of {:d} validation images'.format(nvalid))
    print('There are a total of {:d} images!'.format(nvalid+ntrain+ntest))

    # Build the table
    tbl = Table()
    #print("We have {:d} Traning images [WARNING: SOME ARE FOR TESTS!]".format(ntrain))
    tbl['Type'] = types
    tbl['Date'] = dates
    tbl['Frame'] = frames
    tbl['PI'] = pis
    tbl['set'] = sets
    # Sort
    tbl.sort(['Type', 'Date', 'Frame'])
    # Time
    t = Time(tbl['Date'], out_subfmt='date')

    # Make the LateX Table
    # Open
    tbfil = open(outfil, 'w')

    # tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{deluxetable}{lccccc}\n')
    # tbfil.write('\\rotate\n')
    tbfil.write('\\tablewidth{0pc}\n')
    tbfil.write('\\tablecaption{Training Set\\label{tab:train}}\n')
    tbfil.write('\\tabletypesize{\\small}\n')
    tbfil.write('\\tablehead{\\colhead{Type?} & \\colhead{Date} \n')
    tbfil.write('& \\colhead{Frame} \n')
    tbfil.write('& \\colhead{Use} \n')
    tbfil.write('} \n')

    tbfil.write('\\startdata \n')

    for ii,row in enumerate(tbl):
        if (ii > 15) & sub:
            break

        # Line
        iline = '{:s} & {:s} & {:s} & {:s}'.format(row['Type'], t[ii].value,
                                                   row['Frame'], row['set'])

        # End line
        tbfil.write(iline)
        tbfil.write('\\\\ \n')

    # End Table
    # End
    tbfil.write('\\enddata \n')
    #tbfil.write('\\tablenotetext{a}{Star/galaxy classifier from SExtractor with S/G=1 a star-like object.  Ignored for $\\theta < \\theta_{\\rm min}$.}\n')
    # tbfil.write('\\tablecomments{Units for $C_0$ and $C_1$ are erg/s/cm$^2$/\\AA\ and erg/s/cm$^2$/\\AA$^2$ respecitvely.}\n')
    # End
    tbfil.write('\\end{deluxetable} \n')
    #tbfil.write('\\hline \n')
    #tbfil.write('\\end{tabular} \n')
    #tbfil.write('\\end{minipage} \n')
    #tbfil.write('{$^a$}Restricted to systems with $\mzabs < \mzem$.\\\\ \n')
    #tbfil.write('{$^b$}Quasar is reported to exhibit BAL features by \cite{shen11} (1=True).  We caution that additional BAL features exist in the purported non-BAL quasars.\\\\ \n')
    #tbfil.write('{$^c$}DLA is new (0) or is also reported by N09 (1), PW09 (2), or both (3).\\\\ \n')
    #tbfil.write('\\end{table*} \n')

    #tbfil.write('\\enddata \n')
    #tbfil.write('\\tablenotetext{a}{Flag describing the continuum method applied: 0=Analysis based only on Lyman series lines; 1=Linear fit; 2=Constant fit; 3=Continuum imposed by hand.}\n')
    #tbfil.write('\\tablecomments{Units for $C_0$ and $C_1$ are erg/s/cm$^2$/\\AA\ and erg/s/cm$^2$/\\AA$^2$ respecitvely.}\n')
    # End
    #tbfil.write('\\end{deluxetable*} \n')

    tbfil.close()
    print('Wrote {:s}'.format(outfil))


#### ########################## #########################
def main(flg_tab):

    if flg_tab == 'all':
        flg_tab = np.sum( np.array( [2**ii for ii in range(5)] ))
    else:
        flg_tab = int(flg_tab)

    # DR7 Table
    if flg_tab & (2**0):
        mktab_images(sub=True)# outfil='tab_dr7_dlas_sub.tex', sub=True)
        mktab_images()# outfil='tab_dr7_dlas_sub.tex', sub=True)

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_tab = 0
        flg_tab += 2**0   # Image table
        #flg_tab += 2**1   # DR12
    else:
        flg_tab = sys.argv[1]

    main(flg_tab)
