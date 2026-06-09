#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import uproot
import numpy as np
import mplhep as mh
from matplotlib import gridspec
from scipy.stats import norm

#########################################
file        = uproot.open("jPsiResonance_Np10k.root")
figFilename = 'overlap_Np10k.png'
normalized  = True  
#########################################


def fitBulk(title, data, mean, std, n_bins=100):
        cut = 2*std
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

def computeMeanStd(centers, vals):
        mean = np.average(centers, weights=vals)
        std = np.sqrt(np.average((centers - mean)**2, weights=vals))
        return mean, std
    
# Stile CMS
mh.style.use("CMS")
m_true = 3.0969

def refitAndPlot(h, title, Fcolor, Ecolor):
        vals, bins = h.to_numpy()
        centers = 0.5 * (bins[:-1] + bins[1:]) / m_true if normalized else  0.5 * (bins[:-1] + bins[1:])
        edges = h.axis().edges() / m_true if normalized else h.axis().edges()
        
        m, s = computeMeanStd(centers, vals)
        
        data = np.repeat(centers, h.values().astype(int))
        x, y, mu, sigma = fitBulk(title, data, m, s, n_bins = 100) 
        
        mh.histplot(h.values(), edges,
                    label = rf'$\mathbf{{{title}}}$' + f'\n$\mu$={mu:.2e}\n$\sigma$={sigma:.3e}',
                    histtype='bar',
                    linewidth=2,
                    facecolor=Fcolor,
                    edgecolor=Ecolor)
        # plt.plot(x, y, color=Ecolor, linestyle='-', linewidth=2)
        

if __name__ == '__main__':
        
        h_cmssw = file["tracksACTSRefit/h_mass_cmssw"]
        h_acts = file["tracksACTSRefit/h_mass_acts"]

        plt.figure()
        mh.cms.label("Preliminary", data = False, lumi=None, com='13', loc = 0) 
        refitAndPlot(h_acts, "ACTS", (0.4, 0.6, 1.0, 0.3), (0.4, 0.6, 1.0, 1))
        refitAndPlot(h_cmssw, "CMSSW", (1, 0.5, 0, 0.3), (1, 0.5, 0, 1))
        if normalized:
                plt.xlabel(r"$M_{\mu\mu}$ / $M_{J/\Psi}$")
        else:
                plt.xlabel(r"$M_{\mu\mu}$ [GeV]")
        plt.ylabel("Entries")
        plt.yscale('log')

        plt.legend()

        plt.savefig(f"{figFilename}")
        