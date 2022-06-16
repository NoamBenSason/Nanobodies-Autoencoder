
import tensorflow as tf
import numpy as np
file_models = ["../Best_models/distinctive_multiheaded_attn",
               "../Best_models/efficient_original",
               "../Best_models/prime_attntion"]
TEST_DATA_MAT_LOC = "../NbTestSet"

def main():
    models = [tf.keras.models.load_model(file_model) for file_model in
              file_models]
    mat_coords = np.load(f"{TEST_DATA_MAT_LOC}/test_coord_mat.npy")
    ohe = np.load(f"{TEST_DATA_MAT_LOC}/test_OHE.npy")
    cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                       reduction = tf.keras.losses.Reduction.NONE
                                                       )
    mse_loss = tf.keras.losses.MeanSquaredError(
                                                       reduction = tf.keras.losses.Reduction.NONE
    )
    for model in models:
        pred_mat , pred_seq = model.predict(mat_coords)
        mse_loss_vec = tf.reduce_mean(mse_loss(mat_coords,pred_mat),axis=1)
        cce_loss_vec = tf.reduce_mean(cce_loss(ohe,pred_seq),axis=1)
        print(mse_loss_vec)
        print(cce_loss_vec)






if __name__ == '__main__':
    main()