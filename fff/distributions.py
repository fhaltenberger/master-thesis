import torch
import numpy as np

def moons_distribution(components=100, noise=0.1):
    outer_circ_x = np.cos(np.linspace(0, np.pi, components))
    outer_circ_y = np.sin(np.linspace(0, np.pi, components))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, components))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, components)) - .5

    loc = np.vstack((np.append(outer_circ_x, inner_circ_x),
                    np.append(outer_circ_y, inner_circ_y))).T
    loc = torch.from_numpy(loc)

    normal = torch.distributions.normal.Normal(loc, noise*torch.ones((loc.shape)))
    independent_normal = torch.distributions.independent.Independent(normal, 1)
    mixture = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(torch.ones(200, )),
                                                    independent_normal)
    return mixture