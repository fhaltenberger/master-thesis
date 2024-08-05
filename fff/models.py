import torch
import torch.nn as nn

device ="cpu"

class resnet(nn.Module):
  def __init__(self, n_dim, hidden_dim, n_blocks, output_dim):
    super().__init__()
    self.n_dim = n_dim
    self.n_blocks = n_blocks
    self.output_dim = output_dim
    self.blocks = nn.ModuleList()

    self.first_layer = nn.Linear(n_dim, hidden_dim)

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