import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils
from utils import matrix_to_pdb
from encoder_ver1 import build_encoder
from decoder_ex4 import build_decoder


def plot_val_train_loss(history):
    """
    plots the train and validation loss of the model at each epoch, saves it in 'model_loss_history.png'
    :param history: history object (output of fit function)
    :return: None
    """
    ig, axes = plt.subplots(1, 1, figsize=(15, 3))
    axes.plot(history.history['loss'], label='Training loss')
    axes.plot(history.history['val_loss'], label='Validation loss')
    axes.legend()
    axes.set_title("Train and Val MSE loss")

    plt.savefig(f"/content/drive/MyDrive/ColabNotebooks/model_loss_history.png")


def hardmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_argmax = np.argmax(x, axis=axis)
    y = np.zeros_like(x)
    np.put_along_axis(y, np.expand_dims(x_argmax, axis=axis), 1, axis=axis)
    return y


def train(config=None):
    # _______________loading the data_______________
    labels = np.load(
        "train_input.npy")  # numpy array of shape (1974,NB_MAX_LENGTH,FEATURE_NUM) - data
    mat_input = np.load(
        "train_labels.npy")  # numpy array of shape (1974,NB_MAX_LENGTH,OUTPUT_SIZE) - labels

    inputs, enc_out = build_encoder()
    dec_out = build_decoder(enc_out)
    model = tf.keras.Model(inputs=inputs, outputs=[dec_out, enc_out])

    my_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,clipnorm=1)
    #
    # model = build_network(config)
    # # _______________compiling______________
    #
    model.compile(optimizer=my_optimizer, loss=['mean_squared_error',
                                                tf.keras.losses.CategoricalCrossentropy(from_logits=True)], loss_weights=[1,0.1])
    #
    # # _____________fitting the model______________
    history = model.fit(mat_input, [mat_input,labels],
                        epochs=50,
                        batch_size=128)

    # test_sample = mat_input[0]
    # test_sample = test_sample[None,:]
    # out1, out2 = model.predict(test_sample)
    # seq = hardmax(out2,axis=1)
    # print(np.mean(seq == labels[0]))
    # matrix_to_pdb()

    test_sample = utils.generate_label("1A2Y_1.pdb")
    test_sample = test_sample[None, :]
    out1, out2 = model.predict(test_sample)
    seq = np.argmax(out2[0], axis=1)
    seq_str, real_seq_ind = utils.generate_ind("1A2Y_1.pdb")
    print(np.mean(seq == real_seq_ind))
    matrix_to_pdb(seq_str, out1[0,:,:], "test1A2Y")

    # plot_val_train_loss(history)
    # tf.keras.models.save_model(model, save_dir + model_name)
    # tf.keras.backend.clear_session()


if __name__ == '__main__':
    train()
