#!/usr/bin/env python3

from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plotCMS(zr):
    # PLOT CMS TO HAVE A REFERENCE:
    file1 = uproot.open("/afs/cern.ch/user/l/ldamenti/CMSSW_16_0_0_pre1/src/ACTSinCMSSW/GeometryBuilder/python/Propagator_test_noB.root") 
    tree1 = file1["propagation_steps"]
    mat_x = tree1["g_x"].array(entry_start=0, entry_stop=10000)
    mat_y = tree1["g_y"].array(entry_start=0, entry_stop=10000)
    mat_z = tree1["g_z"].array(entry_start=0, entry_stop=10000)

    # Flatten arrays if needed (assume they're jagged)
    r_mat = np.sqrt(mat_x**2 + mat_y**2)
    x_mat = mat_x
    y_mat = mat_y
    z_mat = mat_z

    # Convert to flat arrays for plotting
    r_flat = ak.to_numpy(ak.flatten(r_mat))
    x_flat = ak.to_numpy(ak.flatten(x_mat))
    y_flat = ak.to_numpy(ak.flatten(y_mat))
    z_flat = ak.to_numpy(ak.flatten(z_mat))
    
    # crea la maschera booleana
    BarrelOnly = (z_flat >= -1000) & (z_flat <= 1000)

    # Plot base
    #plt.scatter(z_flat, r_flat, s=1, c='blue', label='Acts_Sens+Mat')
    if (zr): plt.hist2d(z_flat, r_flat, bins=500, norm=colors.LogNorm(), cmap='viridis')  
    else: plt.hist2d(x_flat[BarrelOnly], y_flat[BarrelOnly], bins=500, norm=colors.LogNorm(), cmap='viridis')  
    

def parseFile(lines, keyLine):
    r_list = []
    z_list = []

    for line in lines:
        if keyLine in line: 
            X = float(line.strip().split()[5])    
            Y = float(line.strip().split()[6])  
            r_list.append(np.sqrt(X*X + Y*Y))
            z_list.append(float(line.strip().split()[7]))
            
    return r_list, z_list 
            
     
    
if __name__ == '__main__':
    filename = 'verboseMatStudy100GeV.txt'
    keyLine = 'particle position at the interaction: '
    
    with open(filename, 'r') as file:
        content = file.read()
        
    tracks = content.split('################################################################################################')

    plt.figure()
    plotCMS(zr = True)
    for i, track in enumerate(tracks):
        if 'SUMMARY: ' in track: continue #Skipping the last part after the refit
        lines = track.splitlines()
        r, z = parseFile(lines, keyLine)
        plt.scatter(z, r, color = 'red', marker = 'x', s = 10, label = 'Mat Interaction')
        
    plt.xlabel("Z")
    plt.ylabel("R")
    plt.legend()
    plt.title("100 Gev (Barrel Only)")
    plt.savefig("MatInteractions_100GeV.png")
    plt.savefig("MatInteractions_100GeV.pdf")
        
        
    
