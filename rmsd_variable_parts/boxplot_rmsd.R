rm(list=ls())
library(ggplot2)
library(reshape)
setwd("C:/Users/USER/Desktop/Uni/BSC/third_year/second semester/data processing/hackathon")

# create rmsd vectors
orig_rmsd<-read.csv("RMSD_results_orig.csv")
atten_rmsd<-read.csv("RMSD_results_atten.csv")
multi_atten_rmsd<-read.csv("RMSD_results_multi_atten.csv")
orig_rmsd_cdr1<-orig_rmsd$cdr1
orig_rmsd_cdr2<-orig_rmsd$cdr2
orig_rmsd_cdr3<-orig_rmsd$cdr3
atten_rmsd_cdr1<-atten_rmsd$cdr1
atten_rmsd_cdr2<-atten_rmsd$cdr2
atten_rmsd_cdr3<-atten_rmsd$cdr3
multi_atten_rmsd_cdr1<-multi_atten_rmsd$cdr1
multi_atten_rmsd_cdr2<-multi_atten_rmsd$cdr2
multi_atten_rmsd_cdr3<-multi_atten_rmsd$cdr3
# colors
ver1_color<-"#EEF6FC"
ver2_color<-"#B6DBF4"
ver3_color<-"#3199DF"

graph_title<-"RMSD between original and predicted structure using different
autoencoder architectures"
# create graph
boxplot(multi_atten_rmsd_cdr1, orig_rmsd_cdr1, atten_rmsd_cdr1, 
        multi_atten_rmsd_cdr2, orig_rmsd_cdr2, atten_rmsd_cdr2,
        multi_atten_rmsd_cdr3, orig_rmsd_cdr3, atten_rmsd_cdr3, main=graph_title, 
        xlab = "architecture index", ylab = "RMSD value",
        names=c("multi headed attention cdr1", "original cdr1", 
                "attention cdr1", "multi headed attention cdr2", 
                "original cdr2", "attention cdr2",
                "multi headed attention cdr3", "original cdr3",
                "attention cdr3"),
        col=c(ver1_color, ver2_color, ver3_color, 
              ver1_color, ver2_color, ver3_color,
              ver1_color, ver2_color, ver3_color))