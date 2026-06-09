#!/usr/bin/env python3

import uproot
import matplotlib.pyplot as plt
import numpy as np
import os
import mplhep as mh
from scipy.stats import norm

mh.style.use("CMS")

if __name__ == '__main__':

    # Percorso del file ROOT
    file_path = '/eos/user/l/ldamenti/DQMfiles/'

    # filename = 'DQMfile_Np10k_E10GeV_muNeg_MatFixed.root'
    # filename = 'DQMfile_Np10k_E10GeV_muNeg_MatFixed_EConly.root'
    # filename = 'DQMfile_effTest_Np1k_E5GeV_muNeg.root'
    filename = 'DQMfile_JacobianTest_onlyEC.root'

    # Percorso dentro il ROOT file fino ai due istogrammi
    histACTS_path = 'DQMData/Run 1/Tracking/Run summary/ACTSmultiTrackValidatortracksACTSRefit_recoCollACTS_ACTSptAssociation/'
    histCMSSW_path = 'DQMData/Run 1/Tracking/Run summary/CMSSWmultiTrackValidatortracksACTSRefit_recoCollCMSSW_CMSSWptAssociation/'

    hist_list   = ['chi2', 'chi2_prob', 'chi2mean']
    hist_ranges = [[0,10], [0,1], [-3,3]]
    hist_titles = [f'Normalized $\chi^2$', f'Normalized $\chi^2$ probability', f'Mean $\chi^2$ vs $\eta$']

    main_dir = 'CompareChi2'

    for hist_name, hist_range, hist_title in zip(hist_list, hist_ranges, hist_titles):
        # Apri il ROOT file
        with uproot.open(file_path + filename) as root_file:
            # Leggi i due istogrammi
            h_acts = root_file[histACTS_path + hist_name]
            h_cmssw = root_file[histCMSSW_path + hist_name]
            
            # Estrai bin e conteggi
            bins_acts = h_acts.axis().edges()  # array dei bordi dei bin
            counts_acts = h_acts.values()      # array dei conteggi
            bins_cmssw = h_cmssw.axis().edges()
            counts_cmssw = h_cmssw.values()
            
            # Controlla che abbiano lo stesso numero di bin (opzionale)
            if len(counts_acts) != len(counts_cmssw):
                print("Attenzione: i due histogrammi hanno un numero diverso di bin!")

        # Plotting sovrapposto
        #plt.figure(figsize=(8,6))
        # Centro dei bin per matplotlib
        bin_centers_acts = 0.5 * (bins_acts[1:] + bins_acts[:-1])
        bin_centers_cmssw = 0.5 * (bins_cmssw[1:] + bins_cmssw[:-1])

        # mean and std values:
        mean_acts = np.average(bin_centers_acts, weights=counts_acts)
        std_acts = np.sqrt(np.average((bin_centers_acts - mean_acts)**2, weights=counts_acts))

        mean_cmssw = np.average(bin_centers_cmssw, weights=counts_cmssw)
        std_cmssw = np.sqrt(np.average((bin_centers_cmssw - mean_cmssw)**2, weights=counts_cmssw))
        
        fig, ax = plt.subplots()        
        mh.histplot(counts_acts, bins_acts, 
                    label=r'$\bf{ACTS}$' + f'\n$\mu$={mean_acts:.2e}\n$\sigma$={std_acts:.2e}', 
                    ax=ax, histtype='bar', 
                    facecolor=(0.4, 0.6, 1.0, 0.3), 
                    edgecolor=(0.4, 0.6, 1.0, 1), 
                    linewidth=2)
        mh.histplot(counts_cmssw, bins_cmssw, 
                    label=r'$\bf{CMSSW}$' + f'\n$\mu$={mean_cmssw:.2e}\n$\sigma$={std_cmssw:.2e}', 
                    ax=ax, histtype='bar', 
                    facecolor=(1, 0.5, 0, 0.3), 
                    edgecolor=(1, 0.5, 0, 1),
                    linewidth=2)

        ax.legend()
        mh.cms.label("Preliminary", data = False, lumi=None, com='0.1', loc = 0)    
        ax.set_xlim(hist_range[0], hist_range[1])
        parameter = hist_name
        if hist_name == 'chi2mean':
            ax.set_xlabel(f'$\eta$')        
            ax.set_ylabel(f'mean $\chi^2$')
        else:
            ax.set_xlabel(hist_title)        
            ax.set_ylabel("Entries")

        # Define the saving directory
        saveDir = main_dir + '/' + parameter + '/'
        os.makedirs(saveDir, exist_ok=True)

        mh.savelabels(saveDir + f'{hist_name}.pdf')  
        mh.savelabels(saveDir + f'{hist_name}.png')  