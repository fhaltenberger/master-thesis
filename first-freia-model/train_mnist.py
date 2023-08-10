import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import config as c
import plotting as p
from inn_model import *

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../mnist_data', 
                    download=True, 
                    train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                        transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
                        transforms.Resize((8,8))
                    ])),
    batch_size=c.BATCHSIZE, shuffle=True, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../mnist_data', 
                    download=True, 
                    train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                        transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
                        transforms.Resize((8,8))
                    ])),
    batch_size=c.BATCHSIZE, shuffle=True, drop_last=True)

def train(plot_loss=False, plot_loss_dyn=False, make_new_model=False, no_save=False):
    if make_new_model==False:
        inn = load_model(c.DEF_PATH)
    else:
        inn = new_model()
    optimizer = torch.optim.Adam(inn.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True, factor=0.6, min_lr=1e-7)
    
    loss_graph = []
    y_loss_graph = []
    z_loss_graph = []
    ljd_graph = []
    test_accuracy_y = []
    
    for epoch in range(c.N_EPOCHS):
        optimizer.zero_grad()
        
        images, labels = next(iter(train_loader))
        images = images.reshape((c.BATCHSIZE, 64))
        
        z, log_jac_det = inn(images, c=make_cond_input(labels))
        #y, z = y_and_z[..., :c.YDIM], y_and_z[..., c.YDIM:]
           
        #y_loss = nn.functional.cross_entropy(y, labels, reduction="none")
        
        z_loss = 0.5 * torch.sum(z**2, dim=-1)
        
        
        loss = z_loss - log_jac_det# + c.LAMBDA_Y_CROSS_ENTROPY * y_loss
        loss = torch.sum(loss, dim=-1) / loss.shape[-1]
               
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
               
        loss_graph.append(loss.detach().numpy())
        #y_loss_graph.append(torch.mean(y_loss).detach().numpy())
        z_loss_graph.append(torch.mean(z_loss).detach().numpy())
        ljd_graph.append(torch.mean(log_jac_det).detach().numpy())
        
        if epoch % 20 == 0 and plot_loss_dyn:                          
            plt.clf()
            plt.close()
            p.plot_losses(loss_graph, z_loss_graph, ljd_graph)
                        
        if epoch == c.N_EPOCHS - 1:
            if not no_save: save_model(inn, c.DEF_PATH)
            p.plot_losses(loss_graph, z_loss_graph, ljd_graph, save=True)
                        
    if plot_loss:
        plt.plot(loss)
        plt.show()

def main():
    #inn = train(plot_loss_dyn=True, make_new_model=True) 
    visual_test(c.COND)

if __name__ == "__main__":
    main()