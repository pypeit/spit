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

vik_path = '/scratch/citrisdance_viktor/viktor_astroimage/'

ddict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 
    'JUN': 6, 'JUNE': 6, 'JUL': 7, 'JULY': 7, 'AUG': 8, 
    'SEP': 9, 'SEPT': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#sys.path.append(os.path.abspath("../Vetting/py"))
#from vette_dr7 import load_ml_dr7


# Summary table of DR7 DLAs
def mktab_training(outfil='tab_training.tex', sub=False):

    # Scan training images
    train_folder = vik_path + 'orig_images/'
    classes = glob.glob(train_folder + '/*')

    all_images = {}
    for iclass in classes:
        # Use rotated to get only one file per input image
        key = os.path.basename(iclass)
        all_images[key] = glob.glob(iclass + '/*_rotated.png')

    # Generate a table
    keys = list(all_images.keys())
    keys.sort()
    types, dates, frames, pis, flags = [], [], [], [], []
    ntrain = 0
    for key in keys:
        # Numbmer of images
        nimg = len(all_images[key])
        print("We have {:d} images of Type={:s}".format(nimg, key))
        ntrain += nimg
        # Type (Capitalize the first letter)
        ckey = key[0].upper() + key[1:]
        types += [ckey]*nimg
        # Loop me
        for ipath in all_images[key]:
            # Parse
            ifile = os.path.basename(ipath)
            if 'xavier' in ifile:
                pis += ['Prochaska']
                # Find date
                year = ifile[18:22]
                month = ddict[ifile[22:25].upper()]
                day = ifile[25:27]
                dates.append('{:s}-{:d}-{:s}'.format(year,month,day))
                # Frame
                i1 = ifile.find('_rotat')
                frames.append(ifile[32:i1])
            else:
                pis += ['Hsyu']
                # Find date
                i0 = ifile.find('_20')
                year = ifile[i0+1:i0+5]
                try:
                    month = ddict[ifile[:4].upper()]
                except:
                    month = ddict[ifile[:3].upper()]
                    day = ifile[3:i0]
                else:
                    day = ifile[4:i0]
                dates.append('{:s}-{:d}-{:s}'.format(year,month,day))
                # Frame
                i1 = ifile.find('_rotat')
                i1b = ifile.find('_verti')
                if i1b>0:
                    i1 = i1b
                frames.append(ifile[i0+6:i1])

    # Build the table
    tbl = Table()
    print("We have {:d} Traning images [WARNING: SOME ARE FOR TESTS!]".format(ntrain))
    tbl['Type'] = types
    tbl['Date'] = dates
    tbl['Frame'] = frames
    tbl['PI'] = pis
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
    tbfil.write('& \\colhead{Frame} & \\colhead{PI} \n')
    tbfil.write('& \\colhead{Test} \n')
    tbfil.write('} \n')

    tbfil.write('\\startdata \n')

    for ii,row in enumerate(tbl):
        if (ii > 15) & sub:
            break

        # Line
        iline = '{:s} & {:s} & {:s} & {:s}'.format(row['Type'], t[ii].value,
                                           row['Frame'], row['PI']) 

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

    if sub:
        return

    # Some stats for the paper
    gd_conf = ml_dlasurvey.confidence > 0.9
    gd_BAL = np.array(bals) == 0
    gd_z = ml_dlasurvey.zabs < ml_dlasurvey.zem
    new = (np.array(N09) == 0) & (~in_dr5)
    gd_zem = ml_dlasurvey.zem < 3.8
    gd_new = gd_BAL & gd_conf & new & gd_z

    new_dlas = Table()
    new_dlas['PLATE'] = ml_dlasurvey.plate[gd_new]
    new_dlas['FIBER'] = ml_dlasurvey.fiber[gd_new]
    new_dlas['zabs'] = ml_dlasurvey.zabs[gd_new]
    new_dlas['NHI'] =  ml_dlasurvey.NHI[gd_new]
    print("There are {:d} DR7 candidates.".format(ml_dlasurvey.nsys))
    print("There are {:d} DR7 candidates not in BAL with zabs<zem.".format(np.sum(gd_BAL&gd_z)))
    print("There are {:d} good DR7 candidates not in BAL.".format(np.sum(gd_BAL&gd_conf&gd_z)))
    print("There are {:d} good DR7 candidates not in N09, PW09 nor BAL".format(np.sum(gd_new)))
    print("There are {:d} good DR7 candidates not in N09, PW09 nor BAL and with zem<3.8".format(np.sum(gd_new & gd_zem)))

    # Write out
    new_dlas.write("new_DR7_DLAs.ascii", format='ascii.fixed_width', overwrite=True)
    pdb.set_trace()

# Summary table of DR12 DLAs
def mktab_dr12(outfil='tab_dr12_dlas.tex', sub=False):

    # Load DLA
    _, dr12_abs = load_ml_dr12()

    # Cut on DLA
    dlas = dr12_abs['NHI'] >= 20.3
    dr12_dla = dr12_abs[dlas]
    dr12_dla_coords = SkyCoord(ra=dr12_dla['RA'], dec=dr12_dla['DEC'], unit='deg')

    # Load Garnett Table 2 for BALs
    tbl2_garnett_file = os.getenv('HOME')+'/Projects/ML_DLA_results/garnett16/ascii_catalog/table2.dat'
    tbl2_garnett = Table.read(tbl2_garnett_file, format='cds')
    tbl2_garnett_coords = SkyCoord(ra=tbl2_garnett['RAdeg'], dec=tbl2_garnett['DEdeg'], unit='deg')

    # Match and fill BAL flag
    dr12_dla['flg_BAL'] = -1
    idx, d2d, d3d = match_coordinates_sky(dr12_dla_coords, tbl2_garnett_coords, nthneighbor=1)
    in_garnett_bal = d2d < 1*u.arcsec  # Check
    dr12_dla['flg_BAL'][in_garnett_bal] = tbl2_garnett['f_BAL'][idx[in_garnett_bal]]
    print("There are {:d} sightlines in DR12 at z>1.95".format(np.sum(tbl2_garnett['z_QSO']>1.95)))
    print("There are {:d} sightlines in DR12 at zem>2 without BALs".format(np.sum(
        (tbl2_garnett['f_BAL']==0) & (tbl2_garnett['z_QSO']>2.))))

    # Load Garnett
    g16_abs = load_garnett16()
    g16_dlas = g16_abs[g16_abs['log.NHI'] >= 20.3]

    # Match
    dr12_to_g16 = match_boss_catalogs(dr12_dla, g16_dlas)
    matched = dr12_to_g16 >= 0
    g16_idx = dr12_to_g16[matched]
    not_in_g16 = dr12_to_g16 < 0

    # Stats
    high_conf = dr12_dla['conf'] > 0.9
    not_bal = dr12_dla['flg_BAL'] == 0
    zlim = dr12_dla['zabs'] > 2.
    gd_zem = dr12_dla['zabs'] < dr12_dla['zem']
    print("There are {:d} high confidence DLAs in DR12, including BALs".format(np.sum(high_conf)))
    print("There are {:d} z>2 DLAs, zabs<zem in DR12 not in a BAL".format(np.sum(not_bal&zlim&gd_zem)))
    print("There are {:d} high confidence z>2 DLAs in DR12 not in a BAL".format(np.sum(high_conf&not_bal&zlim&gd_zem)))
    print("There are {:d} high quality DLAs not in G16".format(np.sum(high_conf&not_bal&zlim&not_in_g16&gd_zem)))
    ml_not_in_g16 = gd_zem&high_conf&not_bal&zlim&not_in_g16
    wrest_ml_not = (1+dr12_dla['zabs'][ml_not_in_g16])*1215.67 / (1+dr12_dla['zem'][ml_not_in_g16])
    below_lyb = wrest_ml_not < 1025.7
    #pdb.set_trace()

    # Open
    tbfil = open(outfil, 'w')
    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\begin{minipage}{170mm} \n')
    tbfil.write('\\caption{BOSS DR12 DLA CANDIDATES$^a$\\label{tab:dr12}}\n')
    tbfil.write('\\begin{tabular}{lcccccccc}\n')
    tbfil.write('\\hline \n')
    #tbfil.write('\\rotate\n')
    #tbfil.write('\\tablewidth{0pc}\n')
    #tbfil.write('\\tabletypesize{\\small}\n')
    tbfil.write('RA & DEC & Plate & Fiber & \\zabs & \\nhi & Conf. & BAL$^b$ \n')
    tbfil.write('& G16$^c$?')
    tbfil.write('\\\\ \n')
    #tbfil.write('& & & (\AA) & (10$^{-15}$) & & (10$^{-17}$) &  ')
    #tbfil.write('} \n')
    tbfil.write('\\hline \n')

    #tbfil.write('\\startdata \n')

    cnt = 0
    for ii,dla in enumerate(dr12_dla):
        if dla['zabs'] < 2.: # RESTRICTING
            continue
        if dla['zabs'] > dla['zem']: # RESTRICTING
            continue
        if sub and (cnt > 5):
            break
        else:
            cnt += 1
        # Generate line
        dlac = '{:0.4f} & {:0.4f} & {:d} & {:d} & {:0.3f} & {:0.2f} & {:0.2f} & {:d}'.format(
            dla['RA'], dla['DEC'],
            dla['Plate'], dla['Fiber'], dla['zabs'], dla['NHI'], dla['conf'], dla['flg_BAL'])
        # G16
        if matched[ii]:
            dlac += '& 1'
        else:
            dlac += '& 0'
        # End line
        tbfil.write(dlac)
        tbfil.write('\\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    tbfil.write('\\end{minipage} \n')
    #tbfil.write('{$^a$}Rest-frame value.  Error is dominated by uncertainty in $n_e$.\\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('{$^a$}Restricted to systems with $\mzabs < \mzem$ and $\mzabs > 2$.\\\\ \n')
    tbfil.write('{$^b$}Quasar is reported to exhibit BAL features by the BOSS survey.\\\\ \n')
    tbfil.write('{$^c$}DLA is new (0) or reported by G16 (1).\\\\ \n')
    tbfil.write('\\end{table*} \n')

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
        mktab_training(sub=True)# outfil='tab_dr7_dlas_sub.tex', sub=True)

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_tab = 0
        flg_tab += 2**0   # Image table
        #flg_tab += 2**1   # DR12
    else:
        flg_tab = sys.argv[1]

    main(flg_tab)
