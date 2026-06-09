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
            r = math.sqrt(x**2 + y**2)
            
            # if r not in R_list: R_list.append(math.sqrt(x**2 + y**2))
            # if x not in X_list: X_list.append(x)
            # if y not in Y_list: Y_list.append(y)
            # if z not in Z_list: Z_list.append(z)
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
    
subDetRange = {
    # subDet Name    Z, R
    'Pixel' : [[-550, 550],[0, 200]],
    'TIB_TID' : [[-1200, 1200],[200, 550]],
    'TOB' : [[-1200, 1200],[550, 1150]],
    'TEC_neg' : [[-3000, -1200],[0, 1200]],
    'TEC_pos' : [[1200, 3000],[0, 1200]],
}
    
def setRange(subDet):
    
    for key, ranges in subDetRange.items():
        if subDet == key:
            z_range = ranges[0]
            r_range = ranges[1] 
            
            plt.xlim(z_range[0], z_range[1])
            plt.ylim(r_range[0], r_range[1])
            

def getSurfCenter(cand):
    X = 0
    Y = 0
    Z = 0
    R = 0
    
    isFound = False 
    
    with open('../12408.0_SingleMuPt100+2023/DetEl_Info.txt', 'r') as file:
        content = file.read()
    
    for line in content.splitlines():
        if cand in line:
            detId = line.strip().split(';')[1]
            subDet = line.strip().split(';')[2]
            position = line.strip().split(';')[3]
            X = float(position.split()[0])
            Y = float(position.split()[1])
            Z = float(position.split()[2])
            R = math.sqrt(float(position.split()[0])**2 + float(position.split()[1])**2)
            isFound = True
            break
        
    if(not isFound):
        print("ERROR: Surface is NOT sensitive")
                
    return X, Y, Z, R 


def getCandidates(lines):
    
    cand_lists = []
    
    for i, line in enumerate(lines):
        if 'Navigation candidates:' in line:
            N_candidates = int(line.strip().split()[5])
            line_elements = lines[i+1].strip().split()
            this_list = []
            for element in line_elements:
                if 'vol' in element:
                    this_list.append(element)
            cand_lists.append((N_candidates, this_list))

    return cand_lists

'''
From List of candidtes to space point
'''
def fromCandtoSP(cand_lists):
    coordinates = []
    for N, candidates in cand_lists:
        this_coordinates = []
        for candidate in candidates:
            x, y, z, r = getSurfCenter(candidate)
            this_coordinates.append([x, y, z, r])
        coordinates.append(this_coordinates)
            
    return coordinates
    

    
if __name__ == '__main__':
    #filename = 'detIDStudy.txt'
    filename = 'sixFitsForDetID_Study.txt'
    withDirection = False
    withHits = True 
    withIDAnalysis = True
    
    with open(filename, 'r') as file:
        content = file.read()
        
    tracks = content.split('################################################################################################')


    # this_track = 4
    this_tracks = [0, 4]

    plt.figure()
    plotCMS(zr = True)
    for i, track in enumerate(tracks):
        isOk = False
        limitsOk = False
        if i not in this_tracks: continue  # USE THIS TO SELECT A GIVEN TRACK(s)
        if len(track) < 50: break
        lines = track.splitlines()
        if '>>>> Fitted parameters:' in lines: isOk = True
        
        if '>>>>> INNER AND OUTER DETID IDENTIFIED' in lines: limitsOk = True
        
        # Print the candidates
        cand_lists = getCandidates(lines)
        coordinates = fromCandtoSP(cand_lists)
        for candidate_list in coordinates:
            for candidate in candidate_list:
                if candidate == [0, 0, 0, 0]: continue
                plt.scatter(candidate[2], candidate[3], color = 'red') # ZR
             
        # Print the KF steps   
        X_list, Y_list, Z_list, R_list  = getPos_list(lines)
        if limitsOk: plt.scatter(Z_list, R_list, color = 'green')
        else: plt.scatter(Z_list, R_list, facecolors='none', edgecolors='green')
                
        plt.title("Candidates (sensitive only) in red and KF steps in green")
        plt.xlabel('z')
        plt.ylabel('r')
        setRange("Pixel")
        plt.savefig(f"candidateTest_rz_.png")
        
                
            
        
        

        
        
    