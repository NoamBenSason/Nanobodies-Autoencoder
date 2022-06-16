
# !pip install Bio
# !pip install import-ipynb
# !pip install wandb

# from google.colab import drive
# drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
# import matplotlib.pyplot as plt
# import wandb

from datetime import datetime

# so we can import utils notebook (delete if working on Pycharm), you might need to change it to your working directory path
# %cd "/content/drive/MyDrive/ColabNotebooks_short"
# import import_ipynb
import utils

# number of ResNet blocks for the first ResNet and the kernel size.
RESNET_1_BLOCKS = 1
RESNET_1_KERNEL_SIZE = 9
RESNET_1_KERNEL_NUM = 15

# number of ResNet blocks for the second ResNet, dilation list to repeat and the kernel size.

RESNET_2_BLOCKS = 5
RESNET_2_KERNEL_SIZE = 5  # good start may be 3/5
RESNET_2_KERNEL_NUM = 28  # DO NOT MAKE IT 1!
DILATION = [1]
WANTED_M = len(DILATION)  # len of DILATION to be randomize by 'wandb' tool

# percentage of dropout for the dropout layer
DROPOUT = 0.289580549283963  # good start may be 0.1-0.5

# number of epochs, Learning rate and Batch size
EPOCHS = 9
LR = 0.0019210418506367384  # good start may be 0.0001/0.001/0.01
BATCH = 16  # good start may be 32/64/128

def get_time():
    now = datetime.now()
    return now.strftime("%d-%m-%Y__%H-%M-%S")


def resnet_block(input_layer, kernel_size, kernel_num, dialation=1):
    """
    create resnet block for decoder
    :param input_layer: input for layer
    :param kernel_size: kernel size
    :param kernel_num: number of kernel
    :param dialation: dialation for block
    :return: output of block
    """
    bn1 = layers.BatchNormalization()(input_layer)
    conv1d_layer1 = layers.Conv1D(kernel_num, kernel_size, padding='same', activation='relu',
                                  dilation_rate=dialation)(bn1)
    bn2 = layers.BatchNormalization()(conv1d_layer1)
    conv1d_layer2 = layers.Conv1D(kernel_num, kernel_size, padding='same', activation='relu',
                                  dilation_rate=dialation)(bn2)
    return layers.Add()([input_layer, conv1d_layer2])


def resnet_1(input_layer, block_num, kernel_size,
             kernel_num):
    """
    ResNet layer - input -> BatchNormalization -> Conv1D -> Relu -> BatchNormalization -> Conv1D -> Relu -> Add
    :param input_layer: input layer for the ResNet
    :return: last layer of the ResNet
    """
    last_layer_output = input_layer

    for _ in range(block_num):
        last_layer_output = resnet_block(last_layer_output, kernel_size, kernel_num)

    return last_layer_output


def resnet_2(input_layer, block_num, kernel_size,
             kernel_num, dial_lst):
    """
    Dilated ResNet layer - input -> BatchNormalization -> dilated Conv1D -> Relu -> BatchNormalization -> dilated Conv1D -> Relu -> Add
    :param input_layer: input layer for the ResNet
    :return: last layer of the ResNet
    """
    last_layer_output = input_layer

    for _ in range(block_num):
        for d in dial_lst:
            last_layer_output = resnet_block(last_layer_output, kernel_size, kernel_num, d)

    return last_layer_output



def build_decoder(input_layer, config=None):
    """
    builds the neural network architecture as shown in the exercise.
    :return: a Keras Model
    """

    # Conv1D -> shape = (NB_MAX_LENGTH, RESNET_1_KERNEL_NUM)
    conv1d_layer = layers.Conv1D(config['RESNET_1_KERNEL_NUM'], config['RESNET_1_KERNEL_SIZE'],
                                 padding='same')(input_layer)

    # first ResNet -> shape = (NB_MAX_LENGTH, RESNET_1_KERNEL_NUM)
    resnet_layer = resnet_1(conv1d_layer, config['RESNET_1_BLOCKS'], config['RESNET_1_KERNEL_SIZE'],
                            config['RESNET_1_KERNEL_NUM'])

    # Conv1D -> shape = (NB_MAX_LENGTH, RESNET_2_KERNEL_NUM)
    conv1d_layer = layers.Conv1D(config['RESNET_2_KERNEL_NUM'], config['RESNET_2_KERNEL_SIZE'],
                                 padding="same")(resnet_layer)

    # second ResNet -> shape = (NB_MAX_LENGTH, RESNET_2_KERNEL_NUM)
    resnet_layer = resnet_2(conv1d_layer, config['RESNET_2_BLOCKS'], config['RESNET_2_KERNEL_SIZE'],
                            config['RESNET_2_KERNEL_NUM'], config['DILATATION'])

    # _____________________________________ Drop out ___________________________________________


    if config['DROP_OUT_TYPE'] == "vanilla":
        dp = layers.Dropout(config['DROPOUT'])(resnet_layer)
    elif config['DROP_OUT_TYPE'] == "gaussian":
        dp = layers.GaussianDropout(rate=0.05, seed=None)(resnet_layer)
    # ____________________________________________________________________________________________

    conv1d_layer = layers.Conv1D(config['RESNET_2_KERNEL_NUM'] // 2, config['RESNET_2_KERNEL_SIZE'],
                                 padding="same",
                                 activation='elu')(dp)

    dense = layers.Dense(15)(conv1d_layer)
    return dense
