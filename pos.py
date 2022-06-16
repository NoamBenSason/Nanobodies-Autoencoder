import fnmatch
import pandas as pd
from calc_positions import calc
from os import listdir
from os.path import isfile,join

model_list = ["rmsd_variable_parts/attn_out_test_pdbs", "rmsd_variable_parts/multi_attention_out_test_pdbs", "rmsd_variable_parts/original_out_test_pdbs"]
files = fnmatch.filter(listdir("NbTestSet"), "*.pdb")
test_samples_list = [f for f in files]
if __name__ == '__main__':
    df = pd.DataFrame(columns=("model", "sample", "cdr1_start", "cdr1_end", "cdr2_start", "cdr2_end", "cdr3_start", "cdr3_end"))
    i = 0
    for model in model_list:
        for sample in test_samples_list:
            ref_pdb = 'NbTestSet/' + sample
            predicted_pdb = model + '/pred_' + sample
            print(ref_pdb, predicted_pdb)
            cdr1_start,cdr1_end, cdr2_start, cdr2_end, cdr3_start, cdr3_end = calc(ref_pdb, predicted_pdb)
            df.loc[i] = [model, sample, cdr1_start,cdr1_end, cdr2_start, cdr2_end, cdr3_start, cdr3_end]
            i += 1
    df.to_csv("RMSD_results_positions.csv", encoding="utf-8")


