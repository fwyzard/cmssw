#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#filename = '12408.0_SingleMuPt100+2023/fitResutls.txt'
#filename = '12408.0_SingleMuPt100+2023/1kPosMu_res.txt'
#filename = '12408.0_SingleMuPt100+2023/asymmetryTest.txt'
filename = '12408.0_SingleMuPt100+2023/10NegMuons_BarrelOnly_results.txt'




qoverP_acts = []

with open(filename, 'r') as file:
    for line in file:
        # if 'qoverP:' in line:
        #     print(float(line.strip().split()[1]))
        if 'q Over P' in line: 
            val = float(line.strip().split()[4])
            print(val)
            if val < 0: 
                qoverP_acts.append(val)
            
            
mu, sigma = norm.fit(qoverP_acts)

print(np.std(qoverP_acts))

plt.hist(qoverP_acts, bins=3, density=True, alpha=0.6, color='skyblue', edgecolor='black')

x = np.linspace(min(qoverP_acts), max(qoverP_acts), 200)
y = norm.pdf(x, mu, sigma)

plt.plot(x, y, 'r-', linewidth=2, label=f'Fit: μ={mu:.3f}, σ={sigma:.3f}')
plt.legend()

plt.xlabel("q Over P")
plt.ylabel("Entries")
plt.title("10 GeV Muons (1000 Events)")

plt.savefig('qOverP_asymmetryTest.png')

