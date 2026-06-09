#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot
import awkward as ak
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
    # plt.scatter(z_flat, r_flat, s=1, c='black', label='Acts_Sens+Mat')
    # if (zr): plt.hist2d(z_flat, r_flat, bins=500, norm=colors.LogNorm(), cmap='viridis')  
    # else: plt.hist2d(x_flat[BarrelOnly], y_flat[BarrelOnly], bins=500, norm=colors.LogNorm(), cmap='viridis')  
    H, xedges, yedges = np.histogram2d(z_flat, r_flat, bins=500)
    plt.contour(
        xedges[:-1], yedges[:-1], H.T,
        colors='black',
        alpha=0.5
    )


def getPos_and_Bzfilename(filename):
    
    R_list = []
    Z_list = []
    Bz_list = []
    
    with open(filename) as file:
        lines = file.readlines()
        
    for i, line in enumerate(lines):
        X = float(line.strip().split()[0])
        Y = float(line.strip().split()[1])
        R_list.append(np.sqrt(X*X + Y*Y))
        Z_list.append(float(line.strip().split()[2]))
        Bz_list.append(float(line.strip().split()[3]))
        
    return R_list, Z_list, Bz_list
        


if __name__ == '__main__':
    r, z, b = getPos_and_Bzfilename('BfieldMap_fixed.txt')
    plt.figure()
    plotCMS(zr = True)
    sc = plt.scatter(z, r, c = b, cmap='cividis', s = 10)
    
    cbar = plt.colorbar(sc)
    cbar.set_label('B [T]')
    sc.set_clim(3.1 ,3.9)
    # cbar.set_ticks([100, 300, 700])
    # cbar.set_ticklabels(['100', '300', '700'])
    plt.ylim(0,1300)
    plt.xlabel('Z')
    plt.ylabel('R')
    plt.savefig("BfieldMap_fixed.png")
    plt.savefig("BfieldMap_fixed.pdf")
    