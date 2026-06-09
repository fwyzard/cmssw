#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colormaps
import uproot
import awkward as ak
import matplotlib.colors as colors

filename = 'verboseLog_5Tracks_Bfixed.txt'

# First of all, open the file once, and split the info for the different tracks:
with open(filename) as file:
    content = file.read()
tracks = content.split('===== Reco track associated to a Tracking Particle: =====')
# tracks = content.split('=============================================================================') # FOR OLD PROCEDURE ONLY

def parseFile(trackIndex):
    r_list = []
    z_list = []
    p_list = []

    # Get the info for this track:
    for i, track in enumerate(tracks):
        if i == trackIndex:
            lines = track.splitlines()
    
    for line in lines:
        if 'KalmanFitter step at pos:' in line:
            X = float(line.strip().split()[7])
            Y = float(line.strip().split()[8])
            
            r_list.append(np.sqrt(X*X + Y*Y))
            z_list.append(float(line.strip().split()[9]))
            p_list.append(float(line.strip().split()[-1]))
            
    return r_list, z_list, p_list

def checkBfied(trackIndex):
    r_list = []
    z_list = []
    Bz_list = []

    # Get the info for this track:
    for i, track in enumerate(tracks):
        if i == trackIndex:
            lines = track.splitlines()
    
    for line in lines:
        if '[DEBUG] B ACTS: At' in line:
            X = float(line.strip().split()[4])
            Y = float(line.strip().split()[5])
            
            r_list.append(np.sqrt(X*X + Y*Y))
            z_list.append(float(line.strip().split()[6]))
            Bz_list.append(float(line.strip().split()[-1]))
    
    return r_list, z_list, Bz_list
    

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
    
  
def getMinMax(nTracks):
    all_p = []
    for i in range(1, nTracks + 1):
        r, z, p = parseFile(i)
        all_p.extend(p) 
    all_p = np.array(all_p)

    return all_p.min(), all_p.max()

def selectRegion(reg):
    if reg == 'PixelBarrel':
        plt.xlim(-300, 300)
        plt.ylim(0, 200)
    elif reg == 'Barrel':
        plt.xlim(-1100, 1100)
        plt.ylim(0, 1200)
    elif reg == 'TIB':
        plt.xlim(-1100, 1100)
        plt.ylim(200, 550)
    elif reg == 'TOB':
        plt.xlim(-1100, 1100)
        plt.ylim(550, 1200)
        
def printRegions():
    # PixelBarrel
    plt.axvline(x=0, linestyle='--', color='red')
    plt.text(20, 20, 'PXBarrel' , color='red', fontweight='bold')
    plt.axvline(x=200, linestyle='--', color='red')
    # TIB
    plt.axvline(x=225, linestyle='--', color='blue')
    plt.text(300, 20, 'TIB' , color='blue', fontweight='bold')
    plt.axvline(x=550, linestyle='--', color='blue')
    # TOB
    plt.axvline(x=575, linestyle='--', color='green')
    plt.text(700, 20, 'TOB' , color='green', fontweight='bold')
    plt.axvline(x=1100, linestyle='--', color='green')

if __name__ == '__main__':
    
    plt.figure()
    plotCMS(zr=True)
    
    nTracks = 5
    minP, maxP = getMinMax(nTracks)
    norm = LogNorm(vmin=minP, vmax=maxP)
    for i in range(1, nTracks + 1):
        r, z, p = parseFile(i)
        sc = plt.scatter(z, r, c = p, cmap='cividis', s = 10, norm=norm)
    
    cbar = plt.colorbar(sc)
    cbar.set_label('Z (log scale)')
    cbar.set_ticks([100, 300, 700])
    cbar.set_ticklabels(['100', '300', '700'])
    plt.xlabel('Z')
    plt.ylabel('R')
    selectRegion('TIB')
    plt.savefig("pMap_Bfixed.png")
    
    plt.figure()
    for i in range(1, nTracks + 1):
        r, z, p = parseFile(i)
        plt.plot(r[:-5], p[:-5], label = f'Track {i}')
        
    plt.plot([min(r), max(r)], [100, 100], linestyle = '--', label = "P = 100 GeV")
    plt.legend()
    printRegions()
    plt.xlabel("R")
    plt.yscale('log')
    plt.ylabel("P [GeV]")
    plt.savefig("PvsR_Bfixed.png")
    
    
    plt.figure()
    for i in range(1, nTracks + 1):
        r, z, p = parseFile(i)
        plt.plot(z[:-4], p[:-4], label = f'Track {i}')
        
    plt.plot([-900, 900], [100, 100], linestyle = '--', label = "P = 100 GeV")
    plt.legend()
    plt.xlabel("Z")
    plt.yscale('log')
    plt.ylabel("P [GeV]")
    plt.savefig("PvsZ_Bfixed.png")
    
    
    # NOTE: THIS CHECK TRIGGERED A DEEPER ANALYSIS DEVELOPED IN A DIFFERENT DIRECTORY
    # plt.figure()
    # for i in range(1, nTracks + 1):
    #     r, z, Bz = checkBfied(i)
    #     plt.plot(r, Bz, label = f'Track {i}')
    
    # plt.ylabel("B [T]")
    # plt.xlabel("R")
    # plt.legend()
    # plt.savefig("Bcheck.png")

   


