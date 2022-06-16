import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import utils

file_models = ["../Best_models/distinctive_multiheaded_attn",
               "../Best_models/efficient_original",
               "../Best_models/prime_attntion"]
NAMES = ["multi_attention", "original", "attn"]
FOLDER_NAME = "out_test_pdbs"
TEST_DATA_MAT_LOC = "../NbTestSet"


def main():
    models = [tf.keras.models.load_model(file_model) for file_model in
              file_models]
    data_path = "../NbTestSet"
    for pdb in tqdm(os.listdir(data_path)):
        if "csv" in pdb:
            continue
        for name, model in zip(NAMES, models):
            file_path = os.path.join(data_path, pdb)
            coord_mat = utils.generate_label(file_path)[None, :]
            aa, _ = utils.get_seq_aa(file_path, utils.NB_CHAIN_ID)
            pred_mat, _ = model.predict(coord_mat)
            utils.matrix_to_pdb(aa, pred_mat[0, :, :], f"{name}_"
                                                       f"{FOLDER_NAME}/pred_{pdb}".replace(".pdb",""))

if __name__ == '__main__':
    main()
