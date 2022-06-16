# NbTestSet/1dlf.pdb
# NbTestSet/2v17.pdb
# NbTestSet/3lmj.pdb
import tensorflow as tf
import numpy as np
import pandas as pd
import utils

files = ["1dlf", "2v17", "3lmj"]
file_models = ["Best_models/distinctive_multiheaded_attn",
               "Best_models/efficient_original", "Best_models/prime_attntion"]
cols = ["pos"] + list(utils.AA_DICT.keys())

if __name__ == '__main__':
    models = [tf.keras.models.load_model(file_model) for file_model in
              file_models]
    for file in files:
        for i, model in enumerate(models):
            rel_path = f"NbTestSet/{file}.pdb"
            coord_mat = utils.generate_label(rel_path)[None, :]
            aa, _ = utils.get_seq_aa(rel_path, utils.NB_CHAIN_ID)
            _, out = model.predict(coord_mat)
            out = out[0, :len(aa)]
            prob = tf.nn.softmax(out, axis=1)
            new_mat = np.zeros((prob.shape[0], prob.shape[1]+1))
            new_mat[:, 0] = np.arange(len(aa))
            new_mat[:, 1:] = prob
            df = pd.DataFrame(new_mat, columns=cols)
            df["pos"] = df["pos"].astype(np.int32)
            df.to_csv(f"prob_csv_{file}_model_{i}.csv",index=False,columns=cols)
