import torch

XDIM = 28*28
YDIM = 10
ZDIM = XDIM

SUBNET_HIDDEN_DIM = 512

N_BLOCKS = 12

BATCHSIZE = 128
N_EPOCHS = 4000

LR_INIT = 1e-3
LR_RED_FACTOR = 6e-1

ADD_NOISE = False
X_NOISE_LEVEL = 0.005

DEF_PATH = "models/model.pt"
EXPERIMENT_NAME = "imgsize28x28_nozfactor_12blocks_1hidden"
