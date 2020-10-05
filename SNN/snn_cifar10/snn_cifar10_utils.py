import torch
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from torch.nn.modules.utils import _pair
from matplotlib.collections import PathCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, List, Optional, Sized, Dict, Union

plt.ion()

def plot_weights(
    weights: torch.Tensor,
    wmin: Optional[float] = 0,
    wmax: Optional[float] = 1,
    im: Optional[AxesImage] = None,
    figsize: Tuple[int, int] = (5, 5),
    cmap: str = "hot_r",
    save: Optional[str] = None,
) -> AxesImage:
    # language=rst
    """
    Plot a connection weight matrix.

    :param weights: Weight matrix of ``Connection`` object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :param save: file name to save fig, if None = not saving fig.
    :return: ``AxesImage`` for re-drawing the weights plot.
    """
    local_weights = weights.detach().clone().cpu().numpy()
    if save is not None:
        plt.ioff()

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(local_weights, cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_aspect("auto")

        plt.colorbar(im, cax=cax)
        fig.tight_layout()

        a = save.split(".")
        if len(a) == 2:
            save = a[0] + ".1." + a[1]
        else:
            a[1] = "." + str(1 + int(a[1])) + ".png"
            save = a[0] + a[1]

        plt.savefig(save, bbox_inches="tight")

        plt.close(fig)
        plt.ion()
        return im, save
    else:
        if not im:
            fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(local_weights, cmap=cmap, vmin=wmin, vmax=wmax)
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.05)

            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_aspect("auto")

            plt.colorbar(im, cax=cax)
            fig.tight_layout()
        else:
            im.set_data(local_weights)

        return im