import torch

XDIM = 64
YDIM = 10
ZDIM = XDIM

N_BLOCKS = 3

BATCHSIZE = 128
N_EPOCHS = 2000

LR_INIT = 1e-2
LR_RED_FACTOR = 6e-1

COND = torch.Tensor([4,])

#LAMBDA_MAX_LIKELIHOOD = 1
#LAMBDA_Y_CROSS_ENTROPY = 10

#EPOCHS_REDUCE_LR = [800, 1500]

DEF_PATH = "models/model_basic_inn_architecture.pt"