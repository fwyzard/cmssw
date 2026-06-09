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

def getPixelsHits(lines):
    x = []
    y = []
    z = []
    r = []
    for line in lines:
        if 'PIXEL Global hit position: ' in line:
            X = float(line.strip().split()[4])
            Y = float(line.strip().split()[5])
            Z = float(line.strip().split()[6])
            x.append(X)
            y.append(Y) 
            z.append(Z)
            
            r.append(math.sqrt(X**2 + Y**2)) 
    
    return x, y, z, r    

def getInitDir(lines):
    x = y = z = r = 0
    for line in lines:
        if 'Direction (normalized):' in line:
            x = float(line.strip().split()[4])
            y = float(line.strip().split()[7])
            z = float(line.strip().split()[10])
            
            r = math.sqrt(x**2 + y**2)
    return x, y, z, r
    

def getPos_list(lines):
    
    X_list = []
    Y_list = []
    R_list = []
    Z_list = []
    
    for line in lines:
        if 'KalmanFitter step at pos' in line:
            x = float(line.strip().split()[7])
            y = float(line.strip().split()[8])
            z = float(line.strip().split()[9])
            
            R_list.append(math.sqrt(x**2 + y**2))
            X_list.append(x)
            Y_list.append(y)
            Z_list.append(z)
            
    return X_list, Y_list, Z_list, R_list  



def plotCMS(zr):
    # PLOT CMS TO HAVE A REFERENCE:
    file1 = uproot.open("../Propagator_test_noB.root") 
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
     
    
def getLastEntry(List):
    tot_len = len(List)
    return List[tot_len - 1] + 100

def getBeforeHits(filename, track_index): 
    X = []
    Y = []
    Z = []
    R = []
    
    with open(filename) as file:
        context = file.read()
        
    tracks = context.split('-----')
    
    for i, track in enumerate(tracks):
        if i == track_index:
            lines = track.splitlines()
    
    for line in lines:
        if line == "": continue
        if 'CMSSW' in filename:
            detId = line.strip().split(';')[0]
            subDet = line.strip().split(';')[1]
            position = line.strip().split(';')[2].split('(')[1].split(')')[0]
            X.append(float(position.split(',')[0])*10)
            Y.append(float(position.split(',')[1])*10)
            Z.append(float(position.split(',')[2])*10)
            R.append(math.sqrt((float(position.split(',')[0])*10)**2 + (float(position.split(',')[1])*10)**2))
        else: 
            detId = line.strip().split(';')[1] 
            subDet = line.strip().split(';')[2]
            position = line.strip().split(';')[3]
            X.append(float(position.split()[0]))
            Y.append(float(position.split()[1]))
            Z.append(float(position.split()[2]))
            R.append(math.sqrt(float(position.split()[0])**2 + float(position.split()[1])**2))
    
    return X, Y, Z, R
      
def plotBefore_reFit_hits(isRZ = True, track_index = 0):
    # ACTS
    X_acts, Y_acts, Z_acts, R_acts = getBeforeHits('../12408.0_SingleMuPt100+2023/DetEl_ACTSInfo_AfterConversion.txt', track_index)
    # CMSSW
    X_cmssw, Y_cmssw, Z_cmssw, R_cmssw = getBeforeHits('../12408.0_SingleMuPt100+2023/DetEl_CMSSWInfo.txt', track_index)
    
    if(isRZ):
        if X_acts != []: plt.scatter(Z_acts, R_acts, color = 'yellow', s = 0.3, marker='x', label = 'sensitive before reFit (ACTS)')
        if X_cmssw != []: plt.scatter(Z_cmssw, R_cmssw, color = 'orange', s = 0.3, marker='x', label = 'sensitive before reFit (CMSSW)')
    else:
        if X_acts != []: plt.scatter(X_acts, Y_acts, color = 'yellow', s = 0.3, marker='x', label = 'sensitive before reFit (ACTS)')
        if X_cmssw != []: plt.scatter(X_cmssw, Y_cmssw, color = 'orange', s = 0.3, marker='x', label = 'sensitive before reFit (CMSSW)')


def runDetIDAnalysis(lines):
    X = []
    Y = []
    Z = []
    R = []
    
    with open('../12408.0_SingleMuPt100+2023/DetEl_Info.txt', 'r') as file:
        content = file.read()
    
    surfInfo_list = []
    for line in lines:
        if 'Update single surface status for surface:' in line: 
            surfInfo = line.strip().split()[9]
            if surfInfo not in surfInfo_list:
                surfInfo_list.append(surfInfo)
    
    isFound = False
    sensCounter = 0
    for this_surf in surfInfo_list:
        if 'sen' in this_surf:
            print("Sensitive surface found:")
            # Search for this sens surface in the info file:
            for line in content.splitlines():
                if this_surf in line:
                    detId = line.strip().split(';')[1]
                    subDet = line.strip().split(';')[2]
                    position = line.strip().split(';')[3]
                    X.append(float(position.split()[0]))
                    Y.append(float(position.split()[1]))
                    Z.append(float(position.split()[2]))
                    R.append(math.sqrt(float(position.split()[0])**2 + float(position.split()[1])**2))
                    print(">>>>> Surface info: ", detId, " ", subDet, " ", position)
                    isFound = True
                    sensCounter += 1
                    break
                
            if(not isFound):
                print("ERROR: Surface NOT found in info file!")
                
    print(f"\nSummary: found {sensCounter} sensitive surfaces")
    return X, Y, Z, R 
    
if __name__ == '__main__':
    filename = 'sixFitsForDetID_Study.txt'
    withDirection = False
    withHits = True 
    withIDAnalysis = True
    
    with open(filename, 'r') as file:
        content = file.read()
        
    tracks = content.split('################################################################################################')

    # this_track = 4
    #this_tracks = [0, 2, 3] #BAD
    this_tracks = [4, 5] #GOOD

    plt.figure()
    plotCMS(zr = True)
    for i, track in enumerate(tracks[:-1]):
        isOk = False
        limitsOk = False
        if i not in this_tracks: continue  # USE THIS TO SELECT A GIVEN TRACK(s)
        if len(track) < 50: break
        lines = track.splitlines()
        if '>>>> Fitted parameters:' in lines: isOk = True
        
        if '>>>>> INNER AND OUTER DETID IDENTIFIED' in lines: limitsOk = True
        
        X_lis, Y_list, Z_list, R_list  = getPos_list(lines)
        x_dir, y_dir, z_dir, r_dir = getInitDir(lines)
        x_hit, y_hit, z_hit, r_hit = getPixelsHits(lines)
        
        if isOk: 
            if limitsOk:
                plt.scatter(Z_list, R_list, color = 'green', s = 0.3, label = 'KalmanFitter steps')
            else:
                plt.scatter(Z_list, R_list, facecolors='none', edgecolors='green', s = 0.3, label = 'KalmanFitter steps')
        else: 
            plt.scatter(Z_list, R_list, color = 'red', s = 0.3, label = 'KalmanFitter steps')
            
        # Print the pixels hits and the direction only if required:
        if withDirection:
            if x_dir != 0: plt.arrow(0, 0, z_dir*1000, r_dir*1000, head_width=50, head_length=100, fc='black', ec='black')
        if withHits:
            if x_hit != []: plt.scatter(z_hit, r_hit, color = 'blue', s = 5, marker='x')
        if withIDAnalysis:
            print(">>>> TRACK INDEX: ",i)
            plotBefore_reFit_hits(True, i)
            x_detId, y_detId, z_detId, r_detId = runDetIDAnalysis(lines)
            if x_detId != []: plt.scatter(z_detId, r_detId, facecolors='none', edgecolors='red', s = 0.3, label = 'sensitive after reFit')
     
    if withIDAnalysis: 
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), markerscale=10.0)
    plt.xlabel('z')
    plt.ylabel('r')
    #plt.xlim(-1*getLastEntry(Z_list), getLastEntry(Z_list))
    #plt.ylim(0, getLastEntry(R_list))
        
    #plt.savefig(f"NegMuons_BarrelOnly_rz_.png")
    plt.savefig(f"{filename.strip().split('.')[0]}_rz_.pdf")
    plt.savefig(f"{filename.strip().split('.')[0]}_rz_.png")
    
    plt.figure()    
    plotCMS(zr = False)
    for i, track in enumerate(tracks):
        isOk = False
        if i not in this_tracks: continue  # USE THIS TO SELECT A GIVEN TRACK
        if len(track) < 50: break
        lines = track.splitlines()
        if '>>>> Fitted parameters:' in lines: isOk = True
        
        X_list, Y_list, Z_list, R_list  = getPos_list(lines)
        x_dir, y_dir, z_dir, r_dir = getInitDir(lines)
        x_hit, y_hit, z_hit, r_hit = getPixelsHits(lines)
        
        if isOk: 
            plt.scatter(X_list, Y_list, color = 'green', s = 0.3, label = 'KalmanFitter steps')
        else: 
            plt.scatter(X_list, Y_list, color = 'red', s = 0.3, label = 'KalmanFitter steps')
        
        if withDirection:
            if x_dir != 0: plt.arrow(0, 0, z_dir*1000, r_dir*1000, head_width=50, head_length=100, fc='black', ec='black')
        if withHits:
            if x_hit != []: plt.scatter(z_hit, r_hit, color = 'blue', s = 5, marker='x')
        if withIDAnalysis:
            plotBefore_reFit_hits(False, i)
            x_detId, y_detId, z_detId, r_detId = runDetIDAnalysis(lines)
            if x_detId != []: plt.scatter(x_detId, y_detId, s = 0.3, marker='*', facecolors='none', edgecolors='red', label = 'sensitive after reFit')
           
    if withIDAnalysis:    
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), markerscale=10.0)     
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.xlim(-1*getLastEntry(X_list), getLastEntry(X_list))
    #plt.ylim(-1*getLastEntry(Y_list), getLastEntry(Y_list))
        
    plt.savefig(f"{filename.strip().split('.')[0]}_xy_.png")
    #plt.savefig(f"NegMuons_BarrelOnly_xy_.png")
    plt.savefig(f"{filename.strip().split('.')[0]}_xy_.pdf")
        
        
    