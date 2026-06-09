#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import uproot
import numpy as np
import mplhep as mh
from matplotlib import gridspec
from scipy.stats import norm

#########################################
filename    = 'jPsi_data.txt'
figFilename = 'overlap_dataAnalysis.png'
normalized  = True  
#########################################

mh.style.use("CMS")
m_true = 3.0969

def getData():
    data_acts = []
    data_cmssw = []
    
    total = 0       
    inRange = 0 
        
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        acts_mass = float(line.split()[0])
        acts_cmssw = float(line.split()[1])
        total += 1
        if 2.8 < acts_mass < 3.5 and 2.8 < acts_cmssw < 3.5:
            inRange += 1
            data_acts.append(float(line.split()[0]))
            data_cmssw.append(float(line.split()[1]))
    
    return data_acts, data_cmssw, total, inRange

if __name__ == '__main__':
    
    data_acts, data_cmssw, total, inRange = getData()
    
    plt.figure()
    bins = 10
    plt.hist(data_acts, bins=bins,   
            label = 'ACTS',   
            histtype='bar',   
            linewidth=2,  
            facecolor=(0.4, 0.6, 1.0, 0.3),  
            edgecolor=(0.4, 0.6, 1.0, 1))
    plt.hist(data_cmssw, bins=bins, 
            label = 'CMSSW', 
            histtype='bar', 
            linewidth=2, 
            facecolor=(1, 0.5, 0, 0.3),    
            edgecolor=(1, 0.5, 0, 1))
    
    plt.yscale('log')
    
    plt.legend()
    plt.savefig(figFilename)
    
    print(f'Total: {total} vs In range {inRange} -> {(inRange / total)*100}%')
        
       