# ____________________________________If we are on ipynb_____________________________________
# !pip install Bio
# !pip install import-ipynb
# !pip install wandb


# from google.colab import drive
# drive.mount('/content/drive')
# import import_ipynb
# so we can import utils notebook (delete if working on Pycharm), you might need to change
# it to your working directory path
# %cd "/content/drive/MyDrive/ColabNotebooks_short"
# __________________________________________________________________________________________________
import tensorflow as tf
from tensorflow.keras import layers
import utils


def resnet_block(input_layer, kernel_size, kernel_num, leaky_alpha, dialation):
    # bn1 = layers.BatchNormalization()(input_layer)
    conv1d_layer1 = layers.Conv1D(kernel_num, kernel_size, padding='same',
                                  dilation_rate=dialation)(input_layer)
    leakyRelu1 = layers.LeakyReLU(alpha=leaky_alpha)(conv1d_layer1)
    # bn2 = layers.BatchNormalization()(conv1d_layer1)
    conv1d_layer2 = layers.Conv1D(kernel_num, kernel_size, padding='same',
                                  dilation_rate=dialation)(leakyRelu1)
    leakyRelu2 = layers.LeakyReLU(alpha=leaky_alpha)(conv1d_layer2)
    return layers.Add()([input_layer, leakyRelu2])


def resnet_1(input_layer, block_num, kernel_size,
             kernel_num, leaky_alpha):
    """
    ResNet layer - input -> BatchNormalization -> Conv1D -> Relu -> BatchNormalization -> Conv1D -> Relu -> Add
    :param input_layer: input layer for the ResNet
    :return: last layer of the ResNet
    """
    last_layer_output = input_layer

    for i in range(block_num):
        last_layer_output = resnet_block(last_layer_output, kernel_size, kernel_num, leaky_alpha)

    return last_layer_output


def resnet_2(input_layer, block_num, kernel_size,kernel_num, dial_lst, leaky_alpha):
    """
    Dilated ResNet layer - input -> BatchNormalization -> dilated Conv1D -> Relu -> BatchNormalization -> dilated Conv1D -> Relu -> Add
    :param input_layer: input layer for the ResNet
    :return: last layer of the ResNet
    """
    last_layer_output = input_layer

    for i in range(block_num):
        for d in dial_lst:
            last_layer_output = resnet_block(last_layer_output, kernel_size, kernel_num, leaky_alpha, d)

    return last_layer_output





def build_encoder(config=None):
    """
    builds the neural network architecture as shown in the exercise.
    :return: a Keras Model
    """
    # input, shape (NB_MAX_LENGTH,FEATURE_NUM)
    input_layer = tf.keras.Input(shape=(utils.NB_MAX_LENGTH, utils.OUTPUT_SIZE))

    # Conv1D -> shape = (NB_MAX_LENGTH, RESNET_1_KERNEL_NUM)
    conv1d_layer = layers.Conv1D(config['RESNET_1_KERNEL_NUM'], config['RESNET_1_KERNEL_SIZE'],
                                 padding='same')(input_layer)

    # first ResNet -> shape = (NB_MAX_LENGTH, RESNET_1_KERNEL_NUM)
    resnet_layer = resnet_1(conv1d_layer, config['RESNET_1_BLOCKS'], config['RESNET_1_KERNEL_SIZE'],
                            config['RESNET_1_KERNEL_NUM'], config['LEAKY_ALPHA'])

    # Conv1D -> shape = (NB_MAX_LENGTH, RESNET_2_KERNEL_NUM)
    conv1d_layer = layers.Conv1D(config['RESNET_2_KERNEL_NUM'], config['RESNET_2_KERNEL_SIZE'],
                                 padding="same")(resnet_layer)

    # second ResNet -> shape = (NB_MAX_LENGTH, RESNET_2_KERNEL_NUM)
    resnet_layer = resnet_2(conv1d_layer, config['RESNET_2_BLOCKS'], config['RESNET_2_KERNEL_SIZE'],
                            config['RESNET_2_KERNEL_NUM'], config['DILATATION'], config['LEAKY_ALPHA'])

    # _____________________________________ Drop out ___________________________________________

    if config['DROP_OUT_TYPE'] == "vanilla":
        dp = layers.Dropout(config['DROPOUT'])(resnet_layer)
    elif config['DROP_OUT_TYPE'] == "gaussian":
        dp = layers.GaussianDropout(rate=0.05, seed=None)(resnet_layer)
    # ____________________________________________________________________________________________

    conv1d_layer = layers.Conv1D(config['RESNET_2_KERNEL_NUM'] // 2, config['RESNET_2_KERNEL_SIZE'],
                                 padding="same")(dp)
    # _______________________________________________________________

    dense = layers.Dense(utils.FEATURE_NUM, name="seq_dense")(conv1d_layer)

    return input_layer, dense


