import torch.nn as nn
import torch
import math
from scipy.stats import special_ortho_group
import numpy as np
import copy
import tqdm
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import distributions as d
import evaluation as e

def make_regressor(n_dim, hidden_dim, n_layers): 
    mle_layers = [nn.Linear(n_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers):
        mle_layers.append(nn.Linear(hidden_dim, hidden_dim))
        mle_layers.append(nn.ReLU())
    mle_layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*mle_layers)


#### INN ###########################################################################

def subnet_constructor(input_size, hidden_size, output_size, dropout, n_hidden_layers):
    layers = nn.Sequential()
    layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(input_size, hidden_size))    
    layers.append(nn.PReLU())
    for _ in range(n_hidden_layers):
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.PReLU())
    layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden_size, output_size))
    
    return layers

def ortogonal_matrix(dim):
    Q = special_ortho_group.rvs(dim)
    return torch.Tensor(Q)

def train_inn(model, batchsize=1000, epochs=20, lr=0.001, track_kl=True,
              calculate_mmd=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    metrics = {}
    loss_history = []
    mmd_history = []    
    mmd_epochs = []
    kl_history = []
    kl_epochs = []
    model_screenshots = []
    
    for epoch in tqdm.tqdm(range(epochs)):
        optimizer.zero_grad()
        model.train()
        
        x, _ = make_moons(n_samples=batchsize, shuffle=True, noise=0.1)        
        scaler = StandardScaler()
        x_normalized = scaler.fit_transform(x)
        x = torch.tensor(x_normalized, dtype=torch.float32)
        
        z, ljd = model(x)

        loss = torch.sum(0.5*torch.sum(z**2, -1)-ljd) / batchsize
        
        loss.backward()
        loss_history.append(loss.item())
        
        optimizer.step()
        
        
        if epoch%20 == 0:
            model.eval()
            if calculate_mmd:
                mmd_history.append(e.mmd_inverse_multi_quadratic(x, model.sample(batchsize)).item())
                mmd_epochs.append(epoch)   
            if track_kl:   
                moons_dist = d.moons_distribution()
                target = torch.Tensor(moons_dist.score_samples(x))
                kl_history.append(e.generalized_kl(model.logprob(x), target).mean().item())
                kl_epochs.append(epoch)
            model_screenshots.append(copy.deepcopy(model))
    metrics["inn_loss"] = loss_history
    if calculate_mmd: metrics["mmd"] = (mmd_epochs, mmd_history)
    if track_kl: metrics["kl"] = (kl_epochs, kl_history)
    return model, metrics, model_screenshots

class coupling_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, subnet_hidden_layers=1): 
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.split1 = math.floor(self.input_size/2)
        self.split2 = self.input_size - self.split1

        self.subnet = subnet_constructor(self.split1, self.hidden_size, 2*self.split2, dropout, subnet_hidden_layers)

    def forward(self, x, rev=False):
        x1, x2 = x[..., :self.split1], x[..., self.split1:]

        params = self.subnet(x1)
        s, t = params[...,:self.split2], params[...,self.split2:]
        s = torch.tanh(s)
        ljd = torch.sum(s, -1)
        

        if not rev:
            s = torch.exp(s)
            x2 = s*x2 + t
            return torch.cat([x1,x2], -1), ljd
        if rev: 
            s = torch.exp(-s)
            x2 = s * (x2-t)
            return torch.cat([x1,x2], -1)

class realNVP(nn.Module):
    def __init__(self, input_size, hidden_size, n_blocks, dropout=0.0, subnet_hidden_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        
        self.coupling_blocks = nn.ModuleList([coupling_block(input_size, 
                                                             hidden_size, 
                                                             dropout, 
                                                             subnet_hidden_layers=2) 
                                              for _ in range(n_blocks)])
        self.orthogonal_matrices = [ortogonal_matrix(input_size) for _ in range(n_blocks-1)]

    def forward(self, x, rev=False):
        if rev: return self._inverse(x)
        return self._forward(x)
    
    def _forward(self, x):
        ljd = torch.zeros((x.shape[0]))
        for l in range(self.n_blocks-1):
            x, partial_ljd = self.coupling_blocks[l](x)
            ljd += partial_ljd
            x = torch.matmul(x, self.orthogonal_matrices[l])
        x, partial_ljd = self.coupling_blocks[-1](x)
        ljd += partial_ljd
        return x, ljd
    
    def _inverse(self, x):
        for l in range (self.n_blocks-1, 0, -1):
            x = self.coupling_blocks[l](x, rev=True)
            x = torch.matmul(x, self.orthogonal_matrices[l-1].T)
        x = self.coupling_blocks[0](x, rev=True)
        return x      
    
    def sample(self, num_samples):
        z = torch.normal(mean=torch.zeros((num_samples, self.input_size)), std=torch.ones((num_samples, self.input_size)))
        return self._inverse(z)
    
    def logprob(self, x):
        z, ljd = self.forward(x)
        return - 0.5*torch.sum(z**2, -1) + ljd - 0.5 * z.shape[1] * np.log(2*math.pi)
    
    
########### FFF Stuff ###############################################################
    
class resnet(nn.Module):
  def __init__(self, input_dim, hidden_dim, n_blocks, output_dim):
    super().__init__()
    self.n_dim = input_dim
    self.n_blocks = n_blocks
    self.output_dim = output_dim
    self.blocks = nn.ModuleList()

    self.first_layer = nn.Linear(input_dim, hidden_dim)

    for _ in range(self.n_blocks):
      self.blocks.append(
          nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))

    self.last_layer = nn.Linear(hidden_dim, output_dim)

  def forward(self, input):
    input = self.first_layer(input)
    for b in range(self.n_blocks):
      residual = self.blocks[b](input)
      input = input + residual
    return self.last_layer(input)

class SkipConnection(nn.Module):
  def __init__(self, dim_in, dim_out, hidden_dim):
    super().__init__()
    self.inner = torch.nn.Sequential(
        torch.nn.Linear(dim_in, hidden_dim), torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, dim_out)
        ).to(device)
    
  def forward(self, x, *args, **kwargs):
    return x + self.inner(x, *args, **kwargs)
    
