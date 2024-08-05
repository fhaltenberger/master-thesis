import torch
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

def moons_distribution(n_samples=30000, noise=0.1, bandwidth=0.2):
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Fit KDE model
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    return kde

def moons_gmm(components=100, noise=0.1):
    outer_circ_x = np.cos(np.linspace(0, np.pi, components))
    outer_circ_x = (outer_circ_x - 0.5) / .8715
    outer_circ_y = np.sin(np.linspace(0, np.pi, components))
    outer_circ_y = (outer_circ_y - 0.25) / .505
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, components))
    inner_circ_x = (inner_circ_x - 0.5) / .8715
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, components)) - .5
    inner_circ_y = (inner_circ_y - 0.25) / .505

    loc = np.vstack((np.append(outer_circ_x, inner_circ_x),
                    np.append(outer_circ_y, inner_circ_y))).T
    loc = torch.from_numpy(loc)

    normal = torch.distributions.normal.Normal(loc, noise*torch.ones((loc.shape)))
    independent_normal = torch.distributions.independent.Independent(normal, 1)
    mixture = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(torch.ones(2*components, )),
                                                    independent_normal)
    return mixture