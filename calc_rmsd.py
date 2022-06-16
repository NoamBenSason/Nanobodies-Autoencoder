import fnmatch

from var_RMSD import calc
from os import listdir
from os.path import isfile,join

model_list = ["rmsd_variable_parts/attn_out_test_pdbs", "rmsd_variable_parts/multi_attention_out_test_pdbs", "rmsd_variable_parts/original_out_test_pdbs"]
files = fnmatch.filter(listdir("NbTestSet"), "*.pdb")
test_samples_list = [f for f in files]
if __name__ == '__main__':
    for model in model_list:
        for sample in test_samples_list:
            ref_pdb = 'NbTestSet/' + sample
            predicted_pdb = model + '/pred_' + sample
            print(ref_pdb, predicted_pdb)
            calc(ref_pdb, predicted_pdb)
