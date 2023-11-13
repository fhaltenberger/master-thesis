import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import config as c
from train_mnist import make_data_loader

def subnet_fc(dims_in, dims_out):
    dim = c.SUBNET_HIDDEN_DIM
    return nn.Sequential(nn.Linear(dims_in, dim), nn.ReLU(), nn.Linear(dim, dims_out))

def test_model_forward(model, test_loader):
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.reshape((c.BATCHSIZE, c.XDIM))
    
    y_and_z, _ = model(test_images)
    y, _ = y_and_z[..., :c.YDIM], y_and_z[..., c.YDIM:]
    
    pred_y = torch.argmax(y, dim=-1)
    
    pred_correct = pred_y == test_labels
    
    return torch.sum(pred_correct.float()/pred_correct.shape[-1])

def sample_backward(model, cond):
    z_sample = torch.normal(torch.zeros((1, c.ZDIM)), torch.ones((1, c.ZDIM)))
    return model(z_sample, c=make_cond_input(cond), rev=True)

def plot_backward_sample(model, cond):
    x_sample, _ = sample_backward(model, cond)
    x_sample = x_sample.reshape((28,28))
    plt.imshow(x_sample.detach().numpy())
    plt.show()
    
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")
    
def load_model(path):
    model = new_model()
    model.load_state_dict(torch.load(path))
    return model

def new_model():
    model = Ff.SequenceINN(c.XDIM)
    for k in range(c.N_BLOCKS):
        model.append(Fm.AllInOneBlock, cond=0, cond_shape=(c.YDIM,), subnet_constructor=subnet_fc, permute_soft=True)
    return model

def visual_test(cond):
    inn = load_model(c.DEF_PATH)
    plot_backward_sample(inn, cond=cond)

def make_cond_input(labels):
    """
    expects a tensor like (4,) or (8, 6, 7, ..., 8)
    generates a (1, batchsize, YDIM) tensor of one-hot condition
    """
    labels = labels.long()
    one_hots = torch.zeros((10, 10))
    for i in range(10):
        one_hots[i,i] = 1
    condition = one_hots[labels]
    condition = condition.unsqueeze(0)
    return condition.expand((1, condition.shape[1], condition.shape[2]))

def plot_latents(dim=None, batchsize=10000):
    loader = make_data_loader(batchsize=batchsize)
    images, labels = next(iter(loader))
    images = images.reshape((batchsize, c.XDIM))
    inn = load_model(c.DEF_PATH)
    latents, _ = inn(images, c=make_cond_input(labels))
    if dim != None:
        latents = latents[:,dim]
    else:
        latents = torch.sqrt(torch.sum(latents ** 2, axis=1))
    plt.hist(latents.detach().numpy(), bins=100)
    plt.show()

    







