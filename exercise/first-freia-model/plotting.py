import matplotlib.pyplot as plt

import config as c

def plot_losses(l, zl, ljd, save=False):
    fig, ax = plt.subplots()
    
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    
    p1, = ax.plot(l, "k-", label="Total loss")
    p2, = twin1.plot(zl, "b--", label="z norm")
    p3, = twin2.plot(ljd, "g--", label="ljd")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    twin1.set_ylabel("z norm")
    twin2.set_ylabel("ljd")
    
    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin1.set_ylim(0, 50)
    twin2.yaxis.label.set_color(p3.get_color())
    
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis="x", **tkw)
    
    ax.legend(handles=[p1,p2,p3])
    
    plt.pause(0.01)
    plt.show(block=False)
    
    if save: 
        plt.savefig(f"figs/losses/loss_{c.EXPERIMENT_NAME}.png")
        print(f"Saved image to ./figs/losses/loss_{c.EXPERIMENT_NAME}.png")
    