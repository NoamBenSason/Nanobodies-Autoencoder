"""
this script creates a prob matrix using our models
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import utils

files = ["1dlf", "2v17", "3lmj"]
file_models = ["../Best_models/distinctive_multiheaded_attn",
               "../Best_models/efficient_original",
               "../Best_models/prime_attntion"]
cols = ["pos"] + list(utils.AA_DICT.keys())


def create_df_for_log(aa, prob):
    new_mat = np.zeros((prob.shape[0], prob.shape[1] + 1))
    new_mat[:, 0] = np.arange(len(aa))
    new_mat[:, 1:] = prob
    df = pd.DataFrame(new_mat, columns=cols)
    df["pos"] = df["pos"].astype(np.int32)
    return df


def main():
    models = [tf.keras.models.load_model(file_model) for file_model in
              file_models]
    for file in files:
        rel_path = f"../NbTestSet/{file}.pdb"
        aa, _ = utils.get_seq_aa(rel_path, utils.NB_CHAIN_ID)
        coord_mat = utils.generate_label(rel_path)[None, :]
        for i, model in enumerate(models):
            _, out = model.predict(coord_mat)
            out = out[0, :len(aa)]
            prob = tf.nn.softmax(out, axis=1)
            df = create_df_for_log(aa, prob)
            df.to_csv(f"prob_csv_{file}_model_{i}.csv", index=False,
                      columns=cols)
        OHE = utils.generate_input(rel_path)
        df = create_df_for_log(aa, OHE[:len(aa)])
        df.to_csv(f"prob_csv_{file}_OHE.csv", index=False, columns=cols)
        break


if __name__ == '__main__':
    main()
