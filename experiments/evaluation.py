import torch
import matplotlib.pyplot as plt
import distributions as d
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

device = "cpu"

def kldiv(model, batchsize=500, encoder=None, normalized=False, cvf=False):
    moons_dist = d.moons_distribution()
    kldiv = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")
    x, _ = make_moons(batchsize, noise=0.1)
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)
    x = torch.Tensor(x_normalized)
    target = torch.Tensor(moons_dist.score_samples(x)).float().to(device)
    if not cvf:
        if encoder is not None:
            z, _ = encoder(x)
            pred = model(z).squeeze()
        else: pred = model(x).squeeze()
    else: pred = model.logprob(x)
    if normalized: return kldiv(pred, target)
    else: return generalized_kl(pred, target)

def generalized_kl(input, target):
    """based on "SBI with generalized KLD, Miller et al 23"""
    return target.exp() * (target - input + input.exp()/target.exp() -1)

def mmd_inverse_multi_quadratic(x, y, bandwidths=None):
    batch_size = x.size()[0]
    # compute the kernel matrices for each combination of x, y
    # (cleverly using broadcasting to do this efficiently)
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    # compute the sum of kernels at different bandwidths
    K, L, P = 0, 0, 0
    if bandwidths is None:
        bandwidths = [0.4, 0.8, 1.6]
    for sigma in bandwidths:
        s = 1.0 / sigma**2
        K += 1.0 / (1.0 + s * (rx.t() + rx - 2.0*xx))
        L += 1.0 / (1.0 + s * (ry.t() + ry - 2.0*yy))
        P += 1.0 / (1.0 + s * (rx.t() + ry - 2.0*xy))

    beta = 1./(batch_size*(batch_size-1)*len(bandwidths))
    gamma = 2./(batch_size**2 * len(bandwidths))
    return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P) 


# PLOTTING
    

def contour_plot(model, ax=None, encoder=None, batchsize=100, cvf=False):    
    """_summary_
    Args:
        model: Model which calculates logprobs (regressor or, if cvf=True, an INN)
        ax (plt.ax, optional): ax for plotting. Defaults to None.
        encoder, optional): Model which turns x to latent z. Defaults to None.
        batchsize (int, optional): Defaults to 100.
        cvf (bool, optional): Whether model is a regressor or an INN. Defaults to False.
    """
    grid = torch.stack(torch.meshgrid(torch.linspace(-2., 2., 100), torch.linspace(-2., 2., 100), indexing='xy'))
    # convert the grid to a batch of 2d points:
    grid = grid.reshape(2, -1).T
    # get the output on all points in the grid
    if not cvf:
        if encoder is None:
            out = torch.exp(model(grid))
        else:
            z, _ = encoder(grid)
            out = torch.exp(model(z))
    if cvf:
        out = torch.exp(model.logprob(grid))
    # plot
    if ax is None:
        cax = plt.matshow(out.detach().numpy().reshape(100,100))
        plt.axis("off")
        cbar = plt.colorbar(cax)
        plt.show()
    else:
        cax = ax.matshow(out.detach().numpy().reshape(100,100))
        ax.axis("off")
        cbar = plt.colorbar(cax)
        
def plot_inn_samples(model, title="Generated samples from INN", axs=None, show_true=False):
    samples = model.sample(1000)
    samples = samples.detach().numpy()
    if axs is None:
        fig, axs = plt.subplots(1,1)
    axs.scatter(samples[:,0], samples[:,1], color="blue", label="sampled", s=6)
    if show_true:
        x, _ = make_moons(1000, noise=0.1, shuffle=True)
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        axs.scatter(x[:,0], x[:,1], color="orange", label="true", s=6)
    axs.set_title(title)
    axs.legend()
        