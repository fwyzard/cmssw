#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters: nMeas	nHoles	O_match	I_match	eta	nHits_cmssw

def plotIOMatch(match, name):
    # Plot 
    fig, ax = plt.subplots(figsize=(12,6))
    width = 0.4  # width of the bars
    x = np.arange(len(match))  # position of the bars 

    ax.set_title(f'{name} detId match VS eta')
    ax.bar(x - width/2, match[0], width=width, color="red", label="Not Match")
    ax.bar(x + width/2, match[1], width=width, color="blue", label="Match")
    
    # X labels
    ax.set_xticks(x)
    ax.set_xticklabels([f"{interval}" for interval in match.index], rotation=45, ha="right")
    ax.set_xlabel("Eta bins")
    ax.set_ylabel("Counts")
    ax.legend()

    plt.tight_layout()
    
    plt.savefig(f"EtaPlots/eta_vs_{name}Match.png")
    

if __name__ == '__main__':
    # Read the file
    pCut = 500
    df = pd.read_csv("data.txt", sep=r"\s+", engine="python").iloc[:pCut]
    
    # ===== nMeas ACTS vs nHits CMSSW =====
    fig, (ax_main, ax_diff) = plt.subplots(2, 1, sharex=True, figsize=(10,8), gridspec_kw={'height_ratios':[3,1]})
    
    ax_main.scatter(df["eta"], df["nMeas"], label = 'nMeas ACTS')
    ax_main.scatter(df["eta"], df["nHits_cmssw"], label = 'nHits CMSSW')
    ax_main.legend()
    
    diff = df["nHits_cmssw"] - df["nMeas"]
    ax_diff.scatter(df["eta"], diff, label='Difference (nCMSSW - nACTS)', color='green')
    ax_diff.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_diff.set_xlabel("Eta")
    ax_diff.set_ylabel("Diff")
    ax_diff.legend()

    plt.tight_layout()
    plt.savefig("EtaPlots/eta_vs_nMeas_nHitsCmssw.png")
    
    # ===== nMeas and nHoles =====
    plt.figure()
    plt.scatter(df["eta"], df["nMeas"], label = 'nMeas ACTS')
    plt.scatter(df["eta"], df["nHoles"], label = 'nHoles ACTS')
    plt.xlabel("Eta")
    plt.legend()
    plt.savefig("EtaPlots/eta_vs_nMeas_nHoles.png")

    # ===== Inner and Outer match analysis =====
    plt.figure()
    # First I create a binned structure for eta:
    etaBins = np.linspace(df["eta"].min(), df["eta"].max(), 11) # -> 10 uniform bins
    # Create a new column using the bins:
    df["etaBins"] = pd.cut(df["eta"], bins=etaBins)
    # Let's count how many 0 and 1 we have per bin:
    outMatch = df.groupby(["etaBins", "O_match"]).size().unstack(fill_value=0)
    inMatch = df.groupby(["etaBins", "I_match"]).size().unstack(fill_value=0)
    # Now plot the two distributions:
    plotIOMatch(outMatch, 'outer')
    plotIOMatch(inMatch, 'inner')
    
    
    
    
    
    
