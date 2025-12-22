import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .candidate import Candidate
from .utils import symlog


def waterfall(
    candidate: Candidate,
    dm: float = 0,
    badchannels: set | list | np.ndarray | None = None,
    datarange: tuple[float, float] = (0.3, 0.7),
    ax=None,
    **options,
):
    if not ax:
        figure = Figure(figsize=(12, 8))
        FigureCanvas(figure)
        ax = figure.add_subplot()

    if badchannels is None:
        badchannels = []

    vmin = options.get("vmin", 0.1)
    vmax = options.get("vmax", 0.9)
    cmap = options.get("cmap", "viridis")
    cbar = options.get("cbar", True)
    xlabel = options.get("xlabel", "samples")
    ylabel = options.get("xlabel", "channels")
    origin = options.get("origin", "upper")
    logscale = options.get("logscale", False)

    badchannels = [128 - value for value in badchannels]

    stokesI = candidate.calc_intensity(dm, badchannels, datarange)

    stokesI = np.ma.filled(stokesI, np.nan)

    if logscale:
        stokesI = symlog(stokesI)

    vmin, vmax = np.nanpercentile(stokesI, (vmin * 100, vmax * 100))

    image = ax.imshow(stokesI.T, aspect="auto", origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    if cbar is True or cbar.lower() == "right":
        cax = divider.append_axes("right", size="5%", pad=0.15)
    elif cbar.lower() == "left":
        cax = divider.append_axes("left", size="5%", pad=0.15)
    ax.figure.colorbar(image, cax=cax, orientation="vertical")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax
