from types import EllipsisType

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .candidate import Candidate
from . import core
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
    ylabel = options.get("ylabel", "channels")
    origin = options.get("origin", "upper")
    logscale = options.get("logscale", False)

    badchannels = [128 - value for value in badchannels]

    stokesI = candidate.calc_intensity(dm, badchannels, datarange)

    stokesI = np.ma.filled(stokesI, np.nan)

    if logscale:
        stokesI = symlog(stokesI)

    vmin, vmax = np.nanpercentile(stokesI, (vmin * 100, vmax * 100))

    image = ax.imshow(
        stokesI.T, aspect="auto", origin=origin, cmap=cmap, vmin=vmin, vmax=vmax
    )

    divider = make_axes_locatable(ax)
    if cbar is True or cbar.lower() == "right":
        cax = divider.append_axes("right", size="5%", pad=0.15)
    elif cbar.lower() == "left":
        cax = divider.append_axes("left", size="5%", pad=0.15)
    ax.figure.colorbar(image, cax=cax, orientation="vertical")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def lightcurve(
    candidate: Candidate,
    dm: float = 0,
    badchannels: set | list | np.ndarray | None = None,
    datarange: tuple[float, float] = (0.3, 0.7),
    ax=None,
    **options,
):
    pass


def background(
    candidate: Candidate,
    dm: float = 0,
    badchannels: set | list | np.ndarray | None = None,
    datarange: tuple[float, float] = (0.3, 0.7),
    ax=None,
    **options,
):
    pass


def bowtie(
    candidate: Candidate,
    dm: tuple[float, float],
    badchannels: set | list | np.ndarray | None = None,
    datarange: tuple[float, float] = (0.3, 0.7),
    ndm: int = 50,
    trange: slice | EllipsisType = Ellipsis,
    ax=None,
    **options,
):
    """
    Create a bowtie plot: varying DM versus time/samples

    Parameters
    ----------
    candidate : Candidate

    dm : tuple[float, float]
        range of the dispersion measure: start and stop

    badchannels : set | list | np.ndarray | None, default=None
        numbers of channels to flag/ignore

    datarange : tuple[float, float], default=(0.3, 0.7)
        Sample fractions within which the burst is located (after
        dedispersion): lower and upper fraction. The remainder area is used for
        background determination

        The background is calculcated once for the central DM (thus for
        ``(dm[0] + dm[1]) / 2``), so the datarange should be for the particular
        DM; all other DM samples use the same background value.

    ndm : int, default=50
        Number of DM samples along the y-axis

    ax : Matplotlib Axes, default=None
        If given, use this axes to draw the graph on
    """

    badchannels = [128 - value for value in badchannels]

    data = core.bowtie(
        candidate.data,
        dm,
        candidate.freqs,
        candidate.header["tsamp"],
        badchannels,
        datarange,
        ndm,
        trange,
    )

    # Calculate the extent for the imshow axes
    if isinstance(trange, EllipsisType):
        extent = [0, len(xx), dm[1], dm[0]]
    else:
        start = trange.start or 0
        stop = trange.stop if trange.stop else len(xx)
        extent = [start, stop, dm[1], dm[0]]

    if not ax:
        figure = Figure(figsize=(12, 8))
        FigureCanvas(figure)
        ax = figure.add_subplot()

    vmin = options.get("vmin", 0.1)
    vmax = options.get("vmax", 0.9)
    cmap = options.get("cmap", "plasma")
    cbar = options.get("cbar", True)
    xlabel = options.get("xlabel", "samples")
    ylabel = options.get("ylabel", "DM")
    origin = options.get("origin", "upper")
    logscale = options.get("logscale", False)

    if logscale:
        data = symlog(data)

    vmin, vmax = np.nanpercentile(data, (vmin * 100, vmax * 100))
    image = ax.imshow(
        data,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("samples")
    ax.set_ylabel("DM")

    return ax


def peaksignal2noise(
    candidate: Candidate,
    dm: float = 0,
    badchannels: set | list | np.ndarray | None = None,
    datarange: tuple[float, float] = (0.3, 0.7),
    ax=None,
    **options,
):
    pass


def grid(
    candidate: Candidate,
    dm: float = 0,
    badchannels: set | list | np.ndarray | None = None,
    datarange: tuple[float, float] = (0.3, 0.7),
    ax=None,
    **options,
):
    pass
