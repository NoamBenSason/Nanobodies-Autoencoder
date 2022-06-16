import numpy as np
import tensorflow as tf
import utils
from utils import matrix_to_pdb
import datetime
from encoder import build_encoder as be1
from encoder_attn import build_encoder as be2
from encoder_mhattn import build_encoder as be3
from decoder import build_decoder
import wandb

ENTITY = "noambs"  # please change to your username when using wandb

VER_DICT = {1: be1, 2: be2, 3: be3}

# __________________________ Defualt Params __________________________
RESNET_1_BLOCKS = 1
RESNET_1_KERNEL_SIZE = 9
RESNET_1_KERNEL_NUM = 15

RESNET_2_BLOCKS = 5
RESNET_2_KERNEL_SIZE = 5
RESNET_2_KERNEL_NUM = 28  # DO NOT MAKE IT 1!
DILATION = [1]
WANTED_M = len(DILATION)  # len of DILATION to be randomize by 'wandb' tool

# percentage of dropout for the dropout layer
DROPOUT = 0.289580549283963

# number of epochs, Learning rate and Batch size
EPOCHS = 50
LR = 0.0019210418506367384
BATCH = 16

LEAKY_ALPHA = 0
LOSS_3D_W = 1
LOSS_SEQ_W = 0.1
CLIP = 1

save_dir = "BestFits/"
model_name = "selected_model2_polar_5"


def get_time():
    now = datetime.time()
    return now.strftime("%d-%m-%Y__%H-%M-%S")


def get_default_config():
    """
    :return: a configuration with the default
    """
    sweep_config = {'RESNET_1_BLOCKS': RESNET_1_BLOCKS,
                    'RESNET_1_KERNEL_SIZE': RESNET_1_KERNEL_SIZE,
                    'RESNET_1_KERNEL_NUM': RESNET_1_KERNEL_NUM,
                    'RESNET_2_BLOCKS': RESNET_2_BLOCKS,
                    'RESNET_2_KERNEL_SIZE': RESNET_2_KERNEL_SIZE,
                    'RESNET_2_KERNEL_NUM': RESNET_2_KERNEL_NUM,
                    'DROPOUT': DROPOUT, 'EPOCHS': EPOCHS, "LR": LR,
                    'DILATATION': DILATION, 'BATCH': BATCH, 'method': 'random',
                    'LEAKY_ALPHA': LEAKY_ALPHA,
                    'LOSS_3D_W': LOSS_3D_W,
                    'LOSS_SEQ_W': LOSS_SEQ_W,
                    'CLIP': CLIP,
                    'DROP_OUT_TYPE': 'vanilla',
                    'metric': {'name': 'loss', 'goal': 'minimize'},
                    'VER': 3,
                    'name': f"BioEx4_{get_time()}"}

    return sweep_config


def get_config():
    sweep_config = {}
    sweep_config['method'] = 'bayes'
    sweep_config['metric'] = {'name': 'best_val_loss', 'goal': 'minimize'}
    sweep_config["early_terminate"] = {
        "type": "hyperband",
        "min_iter": 2,
        "eta": 2,
    }

    sweep_config['name'] = f"BioEx4_{get_time()}"
    param_dict = {
        'RESNET_1_BLOCKS': {'distribution': 'int_uniform', 'min': 1, 'max': 5},
        'RESNET_1_KERNEL_SIZE': {'values': [3, 5, 7, 9]},
        'RESNET_1_KERNEL_NUM': {'value': 15},
        'RESNET_2_BLOCKS': {'distribution': 'int_uniform', 'min': 1, 'max': 5},
        'RESNET_2_KERNEL_SIZE': {'values': [3, 5, 7, 9]},
        'RESNET_2_KERNEL_NUM': {'distribution': 'int_uniform', 'min': 8,
                                'max': 64},
        'DROPOUT': {'distribution': 'uniform', 'min': 0.001, 'max': 0.5},
        'EPOCHS': {'distribution': 'int_uniform', 'min': 40, 'max': 60},
        "LR": {'distribution': 'uniform', 'min': 0.001, 'max': 0.025},
        "leakyAlpha": {'distribution': 'uniform', 'min': 0.001, 'max': 0.1},
        'BATCH': {'values': [16, 32, 64, 128, 256]},
        'LOSS_3D_W': {'distribution': 'uniform', 'min': 0.7, 'max': 1},
        'LOSS_SEQ_W': {'distribution': 'uniform', 'min': 0.1, 'max': 1},
        'DILATATION': {'values': [[1, 2, 4], [1], [1, 2], [1, 4], [1, 2, 4, 8]]},
        'CLIP': {'distribution': 'uniform', 'min': 0.9, 'max': 1.1},
        'DROP_OUT_TYPE': {'values': ["vanilla", "gaussian"]},
        'VER': {'values': [1, 2, 3]}

    }

    sweep_config['parameters'] = param_dict
    return sweep_config


def hardmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_argmax = np.argmax(x, axis=axis)
    y = np.zeros_like(x)
    np.put_along_axis(y, np.expand_dims(x_argmax, axis=axis), 1, axis=axis)
    return y


class WandbCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(WandbCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        wandb.log({'train_loss': logs['loss'], 'train_mse_loss': logs['dense_loss'],
                   'train_seq_loss': logs['seq_dense_loss']})


def train(config=None):
    if config is None:
        config = get_default_config()
    with wandb.init(config=config) as run:
        config = wandb.config

        # _______________loading the data_______________
        labels = np.load(
            "train_input.npy")  # numpy array of shape (1974,NB_MAX_LENGTH,FEATURE_NUM)
        mat_input = np.load(
            "train_labels.npy")  # numpy array of shape (1974,NB_MAX_LENGTH,OUTPUT_SIZE)

        save_dir = "BestFits/"
        model_name = run.name

        # _______________building model_______________
        inputs, enc_out = VER_DICT[config["VER"]](config)
        dec_out = build_decoder(enc_out, config)
        model = tf.keras.Model(inputs=inputs, outputs=[dec_out, enc_out])

        my_optimizer = tf.keras.optimizers.Adam(learning_rate=config['LR'], clipnorm=config['CLIP'])

        # _______________compiling______________

        model.compile(optimizer=my_optimizer, loss=['mean_squared_error',
                                                    tf.keras.losses.CategoricalCrossentropy(
                                                        from_logits=True)],
                      loss_weights=[config['LOSS_3D_W'], config['LOSS_SEQ_W']])

        # _____________creating callbacks_____________

        callbacks_list = [WandbCallback()]

        # _____________fitting the model______________
        history = model.fit(mat_input, [mat_input, labels],
                            epochs=config['EPOCHS'],
                            callbacks=callbacks_list,
                            batch_size=config['BATCH'])

        tf.keras.models.save_model(model, save_dir + model_name)

        tf.keras.backend.clear_session()


if __name__ == '__main__':
    sweep_id = wandb.sweep(get_config(), project="Hackaton", entity=ENTITY)
    wandb.agent(sweep_id, train, count=1000)
    # train()
