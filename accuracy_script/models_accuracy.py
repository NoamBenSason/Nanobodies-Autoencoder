"""
this script is used to calculate accuracy for our models with the test set
the script calculates both Total accuracy and accuracy for each cdr
the script prints them
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import os
import utils

file_models = ["../Best_models/distinctive_multiheaded_attn",
               "../Best_models/efficient_original",
               "../Best_models/prime_attntion"]
NAMES = ["MultiHeadedAttention", "Original", "Attention"]
TEST_DATA_MAT_LOC = "../NbTestSet"


def main():
    models = [tf.keras.models.load_model(file_model) for file_model in
              file_models]
    locations = pd.read_csv("RMSD_results_positions.csv", index_col=0)
    locations["len_cdr_1"] = locations["cdr1_end"] - locations["cdr1_start"] + 1
    locations["len_cdr_2"] = locations["cdr2_end"] - locations["cdr2_start"] + 1
    locations["len_cdr_3"] = locations["cdr3_end"] - locations["cdr3_start"] + 1
    totals_cdr = np.zeros(3)
    totals_cdr[0] = locations["len_cdr_1"].sum()
    totals_cdr[1] = locations["len_cdr_2"].sum()
    totals_cdr[2] = locations["len_cdr_3"].sum()
    counters = np.zeros((3, 3))
    total_cor_for_model = np.zeros(3)
    total = 0
    for pdb in tqdm(os.listdir(TEST_DATA_MAT_LOC)):
        if "csv" in pdb or "npy" in pdb:
            continue
        seq_r, ind = utils.generate_ind(f"{TEST_DATA_MAT_LOC}/{pdb}")
        ind = ind[:len(seq_r)]
        total += len(seq_r)
        coor_mat = utils.generate_label(f"{TEST_DATA_MAT_LOC}/{pdb}")[None, :]
        row = locations.loc[pdb]
        cdr_inds = [(row["cdr1_start"], row["cdr1_end"]),
                    (row["cdr2_start"], row["cdr2_end"]),
                    (row["cdr3_start"], row["cdr3_end"])]
        for i, model in enumerate(models):
            _, pred_seq = model.predict(coor_mat)
            pred_seq = pred_seq[0, :len(seq_r), :]
            pred_seq_ind = np.argmax(pred_seq, axis=1)
            total_cor_for_model[i] += np.sum(ind == pred_seq_ind)
            for j, (start, end) in enumerate(cdr_inds):
                cdr = ind[start:end]
                counters[i, j] += np.sum(cdr == pred_seq_ind[start:end])

    accuracy_cdrs = counters / totals_cdr
    accuracy_total = total_cor_for_model / total

    for i in range(3):
        print(f"{NAMES[i]} total accuracy:{accuracy_total[i]}")
        for j in range(3):
            print(f"cdr{j+1}_accuracy:{accuracy_cdrs[i,j]}")

        print()



    # ind_array = np.array(ind_list)
    # coord_mat = np.load(f"{TEST_DATA_MAT_LOC}/test_coord_mat.npy")
    # for name, model in zip(NAMES, models):
    #     _, pred_seq = model.predict(coord_mat)
    #     pred_seq_ind = np.argmax(pred_seq, axis=-1)
    #     print(f"{name} accuracy:{np.mean(pred_seq_ind == ind_array)}")


if __name__ == '__main__':
    main()
