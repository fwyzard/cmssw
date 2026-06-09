#!/usr/bin/env python3

import uproot
import matplotlib.pyplot as plt
import numpy as np
import os
import mplhep as mh
from scipy.stats import norm

mh.style.use("CMS")

def fitBulk(title, data, mean, std, n_bins=100):
    cut = 2*std
    subset = [i for i in data if mean - cut < i < mean + cut]
    
    mu, sigma = norm.fit(np.array(subset))
    if 'Pull' in title:
        rng = (-10,10)
        print(f"=========== Bulk fit for {title} ===========")
        print(f"Witout the fit: mean = {mean:.2f}; std = {std:.2f}")
        print(f"With the fit:     mu = {mu:.2f}; sigma = {sigma:.2f}")
        print("========================================================")
    else:
        rng = (-0.0015,0.0015)
        print(f"=========== Bulk fit for {title} ===========")
        print(f"Witout the fit: mean = {mean:.2e}; std = {std:.2e}")
        print(f"With the fit:     mu = {mu:.2e}; sigma = {sigma:.2e}")
        print("========================================================")
        
    
    # Not normalized distribution:
    counts, bin_edges = np.histogram(subset, range=rng, bins=n_bins)
    bin_width = bin_edges[1] - bin_edges[0]
    
    # x = np.linspace(min(np.array(subset)), max(np.array(subset)), 1000)
    x = np.linspace(mean - cut, mean + cut, 1000)
    y = norm.pdf(x, mu, sigma) * np.sum(counts * bin_width)
    
    return x, y, mu, sigma

if __name__ == '__main__':

    # Percorso del file ROOT
    file_path = '/eos/user/l/ldamenti/DQMfiles/'

    # filename = 'DQMfile_1kmuNeg_10GeV.root'
    # filename = 'DQMfile_1kmuNeg_10GeVNew.root'
    # filename = 'DQMfile_500muNeg_10GeV.root'
    # filename = 'DQMfile_500muNeg_10GeV_withJ.root'
    # filename = 'DQMfile_Np10k_E10GeV_muNeg_MatFixed.root'
    # filename = 'DQMfile_Np10k_E10GeV_muNeg_MatFixed_EConly.root'
    # filename = 'DQMfile_effTest_Np1k_E5GeV_muNeg.root'
    filename = 'DQMfile_JacobianTest_onlyEC.root'

    # Percorso dentro il ROOT file fino ai due istogrammi
    histACTS_path = 'DQMData/Run 1/Tracking/Run summary/ACTSmultiTrackValidatortracksACTSRefit_recoCollACTS_ACTSptAssociation/'
    histCMSSW_path = 'DQMData/Run 1/Tracking/Run summary/CMSSWmultiTrackValidatortracksACTSRefit_recoCollCMSSW_CMSSWptAssociation/'

    hist_list = ['pullQoverp','pullDxy','pullDz','pullPhi','pullPt','pullTheta']

    main_dir = 'ComparePull'

    for hist_name in hist_list:
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
        
        # Gaussian Refit:
        data_acts = np.repeat(bin_centers_acts, counts_acts.astype(int))
        x_acts, y_acts, mu_pull_acts, sigma_pull_acts = fitBulk(f"ACTS Pull {hist_name}", 
                                                                data_acts, 
                                                                mean_acts, 
                                                                std_acts, 
                                                                n_bins = 100)
        data_cmssw = np.repeat(bin_centers_cmssw, counts_cmssw.astype(int))
        x_cmssw, y_cmssw, mu_pull_cmssw, sigma_pull_cmssw = fitBulk(f"CMSSW Pull {hist_name}", 
                                                                    data_cmssw, 
                                                                    mean_cmssw, 
                                                                    std_cmssw, 
                                                                    n_bins = 100)

        fig, ax = plt.subplots()
        mh.histplot(counts_acts, bins_acts, 
                    label=r'$\bf{ACTS}$' + f'\n$\mu$={mu_pull_acts:.2e}\n$\sigma$={sigma_pull_acts:.2e}', 
                    ax=ax, histtype='bar', 
                    facecolor=(0.4, 0.6, 1.0, 0.3), 
                    edgecolor=(0.4, 0.6, 1.0, 1), 
                    linewidth=2)
        mh.histplot(counts_cmssw, bins_cmssw, 
                    label=r'$\bf{CMSSW}$' + f'\n$\mu$={mu_pull_cmssw:.2e}\n$\sigma$={sigma_pull_cmssw:.2e}', 
                    ax=ax, histtype='bar', 
                    facecolor=(1, 0.5, 0, 0.3), 
                    edgecolor=(1, 0.5, 0, 1),
                    linewidth=2)
        
        # mh.histplot(counts_acts, bins_acts, 
        #             label=r'$\bf{ACTS}$' + f'\n$\mu$={mean_acts:.2e}\n$\sigma$={std_acts:.2e}', 
        #             ax=ax, histtype='bar', 
        #             facecolor=(0.4, 0.6, 1.0, 0.3), 
        #             edgecolor=(0.4, 0.6, 1.0, 1), 
        #             linewidth=2)
        # mh.histplot(counts_cmssw, bins_cmssw, 
        #             label=r'$\bf{CMSSW}$' + f'\n$\mu$={mean_cmssw:.2e}\n$\sigma$={std_cmssw:.2e}', 
        #             ax=ax, histtype='bar', 
        #             facecolor=(1, 0.5, 0, 0.3), 
        #             edgecolor=(1, 0.5, 0, 1),
        #             linewidth=2)
        
        ax.plot(x_acts, y_acts, color=(0.4, 0.6, 1.0, 1), linestyle='-', linewidth=2)
        ax.plot(x_cmssw, y_cmssw, color=(1, 0.5, 0, 1), linestyle='-', linewidth=2)

        ax.legend()
        mh.cms.label("Preliminary", data = False, lumi=None, com='0.1', loc = 0)   
        ax.set_yscale("log")   
        ax.set_xlim(-10, 10)
        parameter = hist_name.replace('pull','')
        ax.set_xlabel(rf"$({parameter}_{{Fit}} - {parameter}_{{True}})/ \sigma_{{Fit}}$")        
        ax.set_ylabel("Entries")

        # Define the saving directory
        saveDir = main_dir + '/' + parameter + '/'
        os.makedirs(saveDir, exist_ok=True)

        mh.savelabels(saveDir + f'{hist_name}.pdf')  
        mh.savelabels(saveDir + f'{hist_name}.png')  