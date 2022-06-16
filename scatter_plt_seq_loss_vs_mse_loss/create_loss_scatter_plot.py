import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

file_models = ["../Best_models/distinctive_multiheaded_attn",
               "../Best_models/efficient_original",
               "../Best_models/prime_attntion"]
TEST_DATA_MAT_LOC = "../NbTestSet"

NAMES = ["MultiHeadedAttention", "Original", "Attention"]


def main():
    models = [tf.keras.models.load_model(file_model) for file_model in
              file_models]
    mat_coords = np.load(f"{TEST_DATA_MAT_LOC}/test_coord_mat.npy")
    ohe = np.load(f"{TEST_DATA_MAT_LOC}/test_OHE.npy")
    cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE
                                                       )
    mse_loss = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE
    )
    for name, model in zip(NAMES, models):
        pred_mat, pred_seq = model.predict(mat_coords)
        mse_loss_vec = tf.reduce_mean(mse_loss(mat_coords, pred_mat), axis=1)
        cce_loss_vec = tf.reduce_mean(cce_loss(ohe, pred_seq), axis=1)
        plt.scatter(mse_loss_vec, cce_loss_vec, alpha=0.5, label=name)
    plt.title("MSE of the structure vs cross entropy of the sequence")
    plt.legend()
    plt.xlabel("MSE of structure")
    plt.ylabel("CCE of sequence")
    plt.savefig("loss_scatter_plot.png")


if __name__ == '__main__':
    main()
