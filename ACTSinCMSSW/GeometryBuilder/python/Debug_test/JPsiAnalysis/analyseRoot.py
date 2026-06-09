#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import uproot
import numpy as np
import mplhep as mh
from matplotlib import gridspec
from scipy.stats import norm
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

import mplhep as mh
mh.style.use("CMS")

#########################################
# filename    = 'jPsiData.root'
filename    = '/eos/user/l/ldamenti/DatasetFarm_files/jPsiData.root'
figFilename = 'overlap_dataAnalysis.png'
normalized  = True  
#########################################

mass_jPsi =  3.096916

# ===== Get all data from the root file =====
file = uproot.open(filename)
tree = file["Events"]

jpsi_mass   = tree["jpsi_mass"].array(library="np")
jpsi_pt     = tree["jpsi_pt"].array(library="np")
jpsi_eta    = tree["jpsi_eta"].array(library="np")
mu1_pt      = tree["mu1_pt"].array(library="np") 
mu1_eta     = tree["mu1_eta"].array(library="np")
mu2_pt      = tree["mu2_pt"].array(library="np")
mu2_eta     = tree["mu2_eta"].array(library="np")
software_id = tree["software_id"].array(library="np")  # 0=CMSSW, 1=ACTS

m1_E = np.sqrt((mu1_pt * np.cosh(mu1_eta))**2 + 0.105**2)
cut      = 50
parToCut = m1_E
jpsi_mass = jpsi_mass[parToCut < cut]
jpsi_pt = jpsi_pt[parToCut < cut]
jpsi_eta = jpsi_eta[parToCut < cut]
mu1_pt = mu1_pt[parToCut < cut]
mu1_eta = mu1_eta[parToCut < cut]
mu2_pt = mu2_pt[parToCut < cut]
mu2_eta = mu2_eta[parToCut < cut]
software_id = software_id[parToCut < cut]

# ===== Debug plot =====
m1_E = np.sqrt((mu1_pt * np.cosh(mu1_eta))**2 + 0.105**2)
plt.figure()
#plt.hist(jpsi_mass[((m1_E < 30) & (m1_E > 25))], bins=50, histtype='step', color='blue')
plt.hist(m1_E[((m1_E < 30) & (m1_E > 25))], bins=50, histtype='step', color='blue')
plt.yscale('log')
plt.savefig('test.png')
    
def filterSW(swID):
    sw_mask = software_id == swID
    return jpsi_mass[sw_mask], jpsi_pt[sw_mask], jpsi_eta[sw_mask], mu1_pt[sw_mask], mu1_eta[sw_mask], mu2_pt[sw_mask], mu2_eta[sw_mask]

def fitBulk(title, data, mean, std, n_bins=100):
    cut = 6*std
    subset = [i for i in data if mean - cut < i < mean + cut]

    mu, sigma = norm.fit(np.array(subset))
    print(f"=========== Bulk fit for {title} ===========")
    print(f"Witout the fit: mean = {mean:.2f}; std = {std:.2f}")
    print(f"With the fit:     mu = {mu:.2f}; sigma = {sigma:.2f}")
    print("========================================================")

    # Not normalized distribution:
    counts, bin_edges = np.histogram(subset, range=(0.9, 1.2), bins=n_bins)
    bin_width = bin_edges[1] - bin_edges[0]

    # x = np.linspace(min(np.array(subset)), max(np.array(subset)), 1000)
    x = np.linspace(mean - cut, mean + cut, 1000)
    y = norm.pdf(x, mu, sigma) * np.sum(counts * bin_width)

    return x, y, mu, sigma
           
def plotMass(swID, bins, faceC, edgeC):
    # ===== Get the data =====
    jpsi_mass, _, _, _, _, _, _= filterSW(swID = swID)
    # ===== Convert the data into an histogram =====
    counts, bin_edges = np.histogram(jpsi_mass, bins=bins, range=(2.9,3.2))
    
    # ===== Perform the refit with the gaussina function =====
    x, y, mu, sigma = fitBulk('ACTS' if swID == 1 else 'CMSSW', jpsi_mass, np.mean(jpsi_mass), np.std(jpsi_mass, ddof=1), n_bins = 100)
    
    # ===== Print the histogram CMS-style =====
    Label = (r'$\bf{ACTS}$' if swID == 1 else r'$\bf{CMSSW}$') + f'\n$\mu$={mu:.4f}\n $\sigma$={sigma:.4f}', 
    mh.histplot(counts, bin_edges, label= Label, histtype='bar', facecolor=faceC, edgecolor=edgeC, linewidth=2)
    
    # ===== Set axis title =====
    plt.ylabel('Entries')
    plt.xlabel(r"$M_{\mu\mu}$ [GeV]")
    
def pltM_vs_EorEta(bins, swID, isMean, isEnergy):
    bins_centers = 0.5 * (bins[:-1] + bins[1:]) 
    
    jpsi_mass, _, _, mu1_pt, mu1_eta, _, _= filterSW(swID = swID)
    
    mean_list = []
    std_list = []
    for i in range(len(bins)-1):
        Min = bins[i]
        Max = bins[i + 1]
        
        if (isEnergy):
            m1_E = np.sqrt((mu1_pt*np.cosh(mu1_eta))**2 + (0.105)**2)  # sqrt(P^2 + m^2) = sqrt((pt*cosh(eta))^2 + m^2)
            mask = (m1_E > Min) & (m1_E < Max)
        else:
            mask = (mu1_eta > Min) & (mu1_eta < Max)
            
        jpsi_filtered = jpsi_mass[mask]
        
        x, y, mu, sigma = fitBulk('ACTS' if swID == 1 else 'CMSSW', jpsi_filtered, np.mean(jpsi_filtered), np.std(jpsi_filtered, ddof=1), n_bins = 100)
        
        if len(jpsi_filtered) == 0:
            mean_list.append(0)
            std_list.append(0)
        else:
            mean_list.append(mu)
            std_list.append(sigma)
            # mean_list.append(np.mean(jpsi_filtered))
            # std_list.append(np.std(jpsi_filtered, ddof=1))
    
    Label = r'$\bf{ACTS}$' if swID == 1 else r'$\bf{CMSSW}$'
    Color = '#1f77b4'if swID == 1 else '#ff7f0e'
    if(isMean):
        plt.scatter(bins_centers, [mean / mass_jPsi for mean in mean_list], label = Label, color = Color)
        plt.plot([bins[0], bins[-1]], [1,1], linestyle = '--', color='black')
        # plt.ylim(0.996, 1.001)
        plt.ylabel(r"$M_{\mu\mu} / M_{J/\psi}$ mean")
    else:
        plt.scatter(bins_centers, std_list, label = Label, color = Color)
        plt.ylabel(r"$M_{\mu\mu}$ std")
        
    plt.xlabel(r"$\mu$ Energy [GeV]" if isEnergy else r"$\eta$ ")
    
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def pltM_vs_EorEtaTest(nbins, swID, isMean, isEnergy):
    """
    Plotta la massa del JPsi in funzione dell'energia o dell'eta
    usando bin equipopolati (quantili) in entrambi i casi.
    """

    # Estrazione dati
    jpsi_mass, _, _, mu1_pt, mu1_eta, _, _ = filterSW(swID=swID)

    # Variabile su cui binniamo
    if isEnergy:
        var = np.sqrt((mu1_pt * np.cosh(mu1_eta))**2 + 0.105**2)  # Energia del muone
        xlabel = r"$\mu$ Energy [GeV]"
    else:
        var = mu1_eta
        xlabel = r"$\eta$"

    # 🔹 Bin equipopolati tramite quantili
    bins = np.quantile(var, np.linspace(0, 1, nbins + 1))
    bins_centers = 0.5 * (bins[:-1] + bins[1:])

    mean_list = []
    std_list = []

    for i in range(len(bins) - 1):
        Min = bins[i]
        Max = bins[i + 1]

        # Maschera sul quantile
        mask = (var > Min) & (var <= Max)
        jpsi_filtered = jpsi_mass[mask]

        if len(jpsi_filtered) == 0:
            mean_list.append(0)
            std_list.append(0)
            continue

        # Fit della massa
        x, y, mu, sigma = fitBulk(
            'ACTS' if swID == 1 else 'CMSSW',
            jpsi_filtered,
            np.mean(jpsi_filtered),
            np.std(jpsi_filtered, ddof=1),
            n_bins=100
        )

        mean_list.append(mu)
        std_list.append(sigma)
        # mean_list.append(np.mean(jpsi_filtered))
        # std_list.append(np.std(jpsi_filtered, ddof=1))

    # Plot style
    Label = r'$\bf{ACTS}$' if swID == 1 else r'$\bf{CMSSW}$'
    Color = '#1f77b4' if swID == 1 else '#ff7f0e'

    if isMean:
        plt.scatter(
            bins_centers,
            np.array(mean_list) / mass_jPsi,
            label=Label,
            color=Color
        )
        plt.plot([bins[0], bins[-1]], [1, 1], '--', color='black')
        plt.ylabel(r"$M_{\mu\mu} / M_{J/\psi}$ mean")
    else:
        plt.scatter(bins_centers, std_list, label=Label, color=Color)
        plt.ylabel(r"$M_{\mu\mu}$ std")

    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(True)
    plt.show()


    
def sciScaleAndSave(title):
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.legend()
    plt.savefig(title)
    
if __name__ == '__main__':
    
    
    # ===== PLOT FULL INVARIANT MASS DISTRIBUTION =====
    bins = 50
    plt.figure()
    mh.cms.label("Preliminary", data = False, lumi=None, com='13', loc = 0) 
    plotMass(swID = 0, bins = bins, faceC = (1, 0.5, 0, 0.3), edgeC = (1, 0.5, 0, 1))         # CMSSW
    plotMass(swID = 1, bins = bins, faceC = (0.4, 0.6, 1.0, 0.3), edgeC = (0.4, 0.6, 1.0, 1)) # ACTS
    plt.yscale('log')
    plt.legend()
    plt.savefig("fullMassDist_2.png")
    
    
    plt.figure()
    E_bins = np.linspace(1, 20, 10)
    nBins = 10
    mh.cms.label("Preliminary", data = False, lumi=None, com='13', loc = 0) 
    pltM_vs_EorEta(bins = E_bins, swID = 0, isMean = True, isEnergy = True) # CMSSW
    pltM_vs_EorEta(bins = E_bins, swID = 1, isMean = True, isEnergy = True) # ACTS
    # pltM_vs_EorEtaTest(nbins = nBins, swID = 0, isMean = True, isEnergy = True) # CMSSW
    # pltM_vs_EorEtaTest(nbins = nBins, swID = 1, isMean = True, isEnergy = True) # ACTS
    sciScaleAndSave("mean_vs_E.png")
    plt.figure()
    mh.cms.label("Preliminary", data = False, lumi=None, com='13', loc = 0) 
    pltM_vs_EorEta(bins = E_bins, swID = 0, isMean = False, isEnergy = True) # CMSSW
    pltM_vs_EorEta(bins = E_bins, swID = 1, isMean = False, isEnergy = True) # ACTS
    # pltM_vs_EorEtaTest(nbins = nBins, swID = 0, isMean = False, isEnergy = True) # CMSSW
    # pltM_vs_EorEtaTest(nbins = nBins, swID = 1, isMean = False, isEnergy = True) # ACTS
    sciScaleAndSave("std_vs_E.png")
    
    plt.figure()
    eta_bins = np.linspace(-2, 2, 15)
    mh.cms.label("Preliminary", data = False, lumi=None, com='13', loc = 0) 
    pltM_vs_EorEta(bins = eta_bins, swID = 0, isMean = True, isEnergy = False) # CMSSW
    pltM_vs_EorEta(bins = eta_bins, swID = 1, isMean = True, isEnergy = False) # ACTS
    # pltM_vs_EorEtaTest(nbins = nBins, swID = 0, isMean = True, isEnergy = False) # CMSSW
    # pltM_vs_EorEtaTest(nbins = nBins, swID = 1, isMean = True, isEnergy = False) # ACTS
    sciScaleAndSave("mean_vs_Eta.png")
    plt.figure()
    mh.cms.label("Preliminary", data = False, lumi=None, com='13', loc = 0) 
    pltM_vs_EorEta(bins = eta_bins, swID = 0, isMean = False, isEnergy = False) # CMSSW
    pltM_vs_EorEta(bins = eta_bins, swID = 1, isMean = False, isEnergy = False) # ACTS
    # pltM_vs_EorEtaTest(nbins = nBins, swID = 0, isMean = False, isEnergy = False) # CMSSW
    # pltM_vs_EorEtaTest(nbins = nBins, swID = 1, isMean = False, isEnergy = False) # ACTS
    sciScaleAndSave("std_vs_Eta.png")
    
    
    

    
    
    
    
    
        