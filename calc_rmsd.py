import fnmatch
import pandas as pd
from var_RMSD import calc
from os import listdir
from os.path import isfile,join

"""
this script calcs the RMSD of the CDRS for our predicted models
"""


model_list = ["rmsd_variable_parts/attn_out_test_pdbs", "rmsd_variable_parts/multi_attention_out_test_pdbs", "rmsd_variable_parts/original_out_test_pdbs"]
files = fnmatch.filter(listdir("NbTestSet"), "*.pdb")
test_samples_list = [f for f in files if "2vxv" not in f and "2w60" not in f]
if __name__ == '__main__':
    df = pd.DataFrame(columns=("model", "sample", "cdr1", "cdr2", "cdr3"))
    i = 0
    for model in model_list:
        a=1
        for sample in test_samples_list:
            ref_pdb = 'NbTestSet/' + sample
            predicted_pdb = model + '/pred_' + sample
            print(ref_pdb, predicted_pdb)
            cdr1, cdr2, cdr3 = calc(ref_pdb, predicted_pdb)
            df.loc[i] = [model, sample, cdr1, cdr2, cdr3]
            i += 1
    df.to_csv("RMSD_results.csv", encoding="utf-8")


