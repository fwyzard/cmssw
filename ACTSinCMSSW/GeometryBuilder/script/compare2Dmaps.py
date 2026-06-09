#!/usr/bin/env python3

import uproot
import matplotlib.pyplot as plt
import numpy as np
import os
import mplhep as mh
from scipy.stats import norm

mh.style.use("CMS")

def getBinCenters(h):
    x_list = []
    y_list = []
    
    # Get borders and bin centers
    x_edges = h.axis(0).edges()
    y_edges = h.axis(1).edges()
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Get bin contents
    values = h.values()  

    # Scan all the bins
    nx, ny = values.shape
    for i in range(nx):
        for j in range(ny):
            if values[i, j] > 0:
                x_list.append(x_centers[i])
                y_list.append(y_centers[j])
    
    return x_list, y_list

if __name__ == '__main__':

    # Percorso del file ROOT
    file_path = '/eos/user/l/ldamenti/DQMfiles/'

    filename = 'DQMfile_JacobianTest_onlyEC.root'

    # Percorso dentro il ROOT file fino ai due istogrammi
    histACTS_path = 'DQMData/Run 1/Tracking/Run summary/ACTSmultiTrackValidatortracksACTSRefit_recoCollACTS_ACTSptAssociation/'
    histCMSSW_path = 'DQMData/Run 1/Tracking/Run summary/CMSSWmultiTrackValidatortracksACTSRefit_recoCollCMSSW_CMSSWptAssociation/'

    hist_list   = ['phires_vs_eta', 'ptres_vs_eta', 'cotThetares_vs_eta', 'dxyres_vs_eta', 'dzres_vs_eta']
    
    # hist_ranges = [[0,10], [0,1], [-3,3]]
    # hist_titles = [f'Normalized $\chi^2$', f'Normalized $\chi^2$ probability', f'Mean $\chi^2$ vs $\eta$']

    main_dir = 'Compare2Dmaps'

    for hist_name in hist_list:
        plt.figure()
        # Apri il ROOT file
        with uproot.open(file_path + filename) as root_file:
            # Leggi i due istogrammi
            h_acts = root_file[histACTS_path + hist_name]
            x_acts, y_acts = getBinCenters(h_acts)
            h_cmssw = root_file[histCMSSW_path + hist_name]
            x_cmssw, y_cmssw = getBinCenters(h_cmssw)
            
        plt.scatter(x_acts,y_acts, label = r'$\bf{ACTS}$')
        plt.scatter(x_cmssw,y_cmssw, label = r'$\bf{CMSSW}$')
        plt.ylabel(hist_name.split('_')[0])
        plt.xlabel(hist_name.split('_')[2])
        
        
        # Define the saving directory
        saveDir = main_dir + '/' + hist_name + '/'
        os.makedirs(saveDir, exist_ok=True)
        
        mh.cms.label("Preliminary")
        plt.savefig(saveDir + f'{hist_name}.png')
        plt.savefig(saveDir + f'{hist_name}.pdf')