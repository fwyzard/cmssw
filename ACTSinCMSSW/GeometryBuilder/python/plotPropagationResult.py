#!/usr/bin/env python3


# DOES NOT WORK. USE plotCMS.cc INSTEAD!

import uproot
import awkward as ak
import numpy as np
from array import array
import ROOT
import argparse
import os
import math

def plot_propagation(root_file, output_prefix="propagation_plot"):
    # Open the ROOT file with uproot
    try:
        file = uproot.open(root_file)
        tree = file["propagation_steps"]
        g_x = ak.to_numpy(ak.flatten(tree["g_x"].array()))
        g_y = ak.to_numpy(ak.flatten(tree["g_y"].array()))
        g_z = ak.to_numpy(ak.flatten(tree["g_z"].array()))
    except Exception as e:
        print(f"Error opening or reading the ROOT file: {e}")
        return

    # Compute r = sqrt(g_x^2 + g_y^2)
    r = np.sqrt(g_x**2 + g_y**2)

    # Convert numpy arrays to array.array for TGraph
    g_z_arr = array("d", g_z)
    r_arr   = array("d", r)

    # Create TGraph
    graph = ROOT.TGraph(len(g_z_arr), g_z_arr, r_arr)
    graph.SetTitle("Propagation Test; g_z; sqrt(g_x^2 + g_y^2)")
    graph.SetMarkerSize(0.5)

    # Draw on canvas
    c = ROOT.TCanvas("c1", "Propagation Plot", 800, 600)
    graph.Draw("COLZ")  # A=axes, P=points, L=line

    # Save output
    c.SaveAs(f"{output_prefix}.png")
    c.SaveAs(f"{output_prefix}.pdf")
    print(f"Plot saved as {output_prefix}.png and {output_prefix}.pdf")

    c.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast plot of propagation_steps from a ROOT file using uproot + ROOT")
    parser.add_argument("root_file", help="ROOT file to read")
    parser.add_argument("--output", default="propagation_plot", help="Prefix for output files")
    args = parser.parse_args()

    if not os.path.isfile(args.root_file):
        print(f"File {args.root_file} not found!")
    else:
        plot_propagation(args.root_file, args.output)
