import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import config as c
import plotting as p
from inn_model import *

def make_data_loader(train=False, batchsize=c.BATCHSIZE):
    return torch.utils.data.DataLoader(
                    datasets.MNIST('../mnist_data', 
                    download=True, 
                    train=train,
                    transform=transforms.Compose([
                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                        #transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
                        transforms.Resize((28,28))
                    ])),
                    batch_size=batchsize, shuffle=True, drop_last=True)

test_loader = make_data_loader(train=False)
train_loader = make_data_loader()
"""test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../mnist_data', 
                    download=True, 
                    train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                        #transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
                        transforms.Resize((28,28))
                    ])),
                    batch_size=c.BATCHSIZE, shuffle=True, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../mnist_data', 
                    download=True, 
                    train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                        #transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
                        transforms.Resize((28,28))
                    ])),
                    batch_size=c.BATCHSIZE, shuffle=True, drop_last=True)"""

def train(plot_loss_dyn=False, make_new_model=False, no_save=False):
    if make_new_model==False:
        inn = load_model(c.DEF_PATH)
    else:
        inn = new_model()
    optimizer = torch.optim.Adam(inn.parameters(), lr=c.LR_INIT)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True, factor=c.LR_RED_FACTOR, min_lr=1e-7)
    
    loss_graph = []
    z_loss_graph = []
    ljd_graph = []
    
    for epoch in tqdm(range(c.N_EPOCHS)):
        optimizer.zero_grad()
        
        images, labels = next(iter(train_loader))
        images = images.reshape((c.BATCHSIZE, c.XDIM))

        if c.ADD_NOISE:
            noise = torch.normal(torch.zeros((c.BATCHSIZE, 64)), c.X_NOISE_LEVEL*torch.ones((c.BATCHSIZE, 64)))
            images = images + noise
       
        z, log_jac_det = inn(images, c=make_cond_input(labels))
        
        z_loss = 0.5 * torch.sum(z**2, dim=-1)
        loss = z_loss - log_jac_det
        loss = torch.mean(loss) / c.XDIM
               
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
               
        loss_graph.append(loss.detach().numpy())
        z_loss_graph.append(torch.mean(z_loss).detach().numpy())
        ljd_graph.append(torch.mean(log_jac_det).detach().numpy())
        
        if epoch % 20 == 0 and plot_loss_dyn:                          
            plt.clf()
            plt.close()
            p.plot_losses(loss_graph, z_loss_graph, ljd_graph)
                        
        if epoch == c.N_EPOCHS - 1:
            if not no_save: save_model(inn, c.DEF_PATH)
            p.plot_losses(loss_graph, z_loss_graph, ljd_graph, save=True)

        if epoch == 0: print("Training started.")

def main():
    train(make_new_model=True) 

if __name__ == "__main__":
    main()