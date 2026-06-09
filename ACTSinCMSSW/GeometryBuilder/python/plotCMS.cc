#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <TH1F.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <TApplication.h>
#include <TFile.h>
#include <TTree.h>

/* Example of input:
    .L PlotMaterial.cc 
    root [1] plotCMS({"Propagator_test.root","propagation_steps","sqrt(g_x*g_x + g_y*g_y):g_z"})
*/

int plotCMS (std::initializer_list<std::string> list){

    std::string filename, TreeName, Expression;
    int i = 0; 

    for (std::string value : list){
        if (i==0) filename = value;
        if (i==1) TreeName = value;
        if (i==2) Expression = value;
        i++;
    }

    TFile *file = TFile::Open(filename.c_str());

    TTree *tree1 = (TTree*)file->Get(TreeName.c_str());
    
    TCanvas *can = new TCanvas("can", "Canvas", 800, 600);

    TH2F *hist = new TH2F("hist", "L1 vs L2", 1000, -3500, 3500, 1000, 0, 1250);
    //TH2F *hist = new TH2F("hist", "L1 vs L2", 1000, -40000, 40000, 1000, 0, 12000);

    std::string hist_n = Expression + ">> hist";

    tree1->Draw(hist_n.c_str());

    can->Update();

    hist->Draw("COLZ");
    hist->SetTitle("Propagation test");
    hist->GetXaxis()->SetTitle("g_z");
    hist->GetYaxis()->SetTitle("sqrt(g_x*g_x + g_y*g_y)");
    //hist->GetXaxis()->SetTitle("g_x");
    //hist->GetYaxis()->SetTitle("g_y");

    // can->SaveAs("Propagation_test_noB.pdf");
    // can->SaveAs("Propagation_test_noB.root");
    //can->SaveAs("MaterialPlot_test.root");

    can->SaveAs("Propagation_test_Bdebug.pdf");
    can->SaveAs("Propagation_test_Bdebug.root");

    return 0;
}
