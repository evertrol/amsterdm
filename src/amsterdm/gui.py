from bokeh.models import PrintfTickFormatter
import holoviews as hv
from holoviews import streams
import hvplot
import hvplot.xarray  # noqa: F401
import numpy as np
import panel as pn
import param
import xarray as xr

from .candidate import openfile
from .utils import symlog10


hv.extension("bokeh")
pn.extension()
hv.config.image_rtol = 1e-1


COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "fire",
    "gray",
]
DM_ZOOM_LEVELS = {
    "Zoom 1x": {"zoom": 1, "step": 5},
    "Zoom 10x": {"zoom": 10, "step": 1},
    "Zoom 100x": {"zoom": 100, "step": 0.1},
    "Zoom 1000x": {"zoom": 1000, "step": 0.01},
}


class CandidatePlot(param.Parameterized):
    dm = param.Number(default=0.0, bounds=(0.0, 2000.0), step=0.1, label="DM")
    dm_zoom = param.Selector(
        objects=DM_ZOOM_LEVELS.keys(), label="DM slider zoom factor"
    )
    colormap = param.Selector(objects=COLORMAPS, default="viridis", label="Colormap")
    cmin = param.Number(default=0.1, bounds=(0.0, 1.0), label="Lower fraction")
    cmax = param.Number(default=0.9, bounds=(0.0, 1.0), label="Upper fraction")
    logimage = param.Boolean(default=False, label="Logarithmic color scale")
    loglc = param.Boolean(default=False, label="Logarithmic light curve y-axis")
    trunclc = param.Number(default=0, label="Lower y limit")
    datarange = param.Range((0.3, 0.6), bounds=(0, 1))
    badchanlist = param.String(default="", label="Bad channels")
    # Simple flag for cases where the plot method needs to be explicitly triggered
    update_data = param.Boolean(default=False)
    update_plot = param.Integer(default=0)

    def __init__(self, candidate, invert=True, width=800, **kwargs):
        super().__init__(**kwargs)
        self.candidate = candidate
        self.tap = streams.Tap(x=1, y=1)
        self.tap.param.watch(self._on_tap, ["y"])
        self.invert = invert
        self.width = width
        self.range_stream = hv.streams.RangeX()

        self.dm_slider = pn.Param(
            self.param.dm, widgets={"dm": pn.widgets.FloatSlider}
        )[0]

        self._init_data()

        self.param.watch(
            self._update_data,
            [
                "dm",
                "datarange",
                "badchanlist",
                "update_data",
                "logimage",
                "loglc",
                "trunclc",
            ],
        )
        self.param.watch(self._update_dm_slider, ["dm_zoom"])

    def _on_tap(self, event):
        # Simply read the current values from the tap stream
        channel = int(self.tap.y + 0.5)
        if channel in self.badchannels:
            self.badchannels.remove(channel)
        else:
            self.badchannels.add(channel)
        self.badchanlist = ",".join(str(value) for value in sorted(self.badchannels))

        self.update_data = True

    def _update_dm_slider(self, event):
        """Update the DM slider range based on the DM slider zoom selector"""
        config = DM_ZOOM_LEVELS[event.new]
        zoom = config["zoom"]
        step = config["step"]
        value = self.dm
        if zoom == 1:
            start = config["start"]
            end = config["end"]
            width = end - start
        else:
            width = 1000 / config["zoom"]
            start = max(0, self.dm - 2 * width)  # ensure no negative values
            end = self.dm + 2 * width
        value = self.dm // step * step
        self.dm_slider.start = start
        self.dm_slider.end = end
        self.dm_slider.step = step
        self.dm_slider.value = value

    def _init_data(self):
        self.badchannels = set()
        self.channels = list(range(1, len(self.candidate.freqs) + 1))
        if self.invert:
            self.channels = self.channels[::-1]
        self.dt = (self.candidate.times - self.candidate.times[0]) * 1000

        self._calc_data()

    def _calc_data(self):
        # dm = {
        #    "dm": self.dm,
        #    "freq": self.candidate.freqs,
        #    "tsamp": self.candidate.header["tsamp"],
        # }
        self.stokesI, self.bkg = self.candidate.calc_intensity(
            self.dm,
            {128 - value for value in self.badchannels},
            self.datarange,
            ...,
            bkg_extra=True,
        )

        self.stokesI = np.ma.filled(
            self.stokesI, np.nan
        )  # Replace the (temporary) mask with NaNs for plotting purposes and `nanpercentile`

        self.lc = np.nansum(self.stokesI, axis=1)
        if self.loglc:
            self.lc = symlog10(self.lc)
        self.lc[self.lc < self.trunclc] = np.nan

        if self.logimage:
            self.stokesI = symlog10(self.stokesI)

    def _update_data(self, event):
        self.update_data = (
            False  # change back, so assignment in _on_tap triggers this method
        )

        if self.badchanlist:
            self.badchannels = {int(value) for value in self.badchanlist.split(",")}
        else:
            self.badchannels = set()

        self._calc_data()

        self.update_plot += 1

    @param.depends("update_plot")
    def plot_lc(self, x_range=None):
        lcplot = hv.Curve((self.dt, self.lc), "t", "I").opts(
            width=self.width, framewise=True
        )
        if x_range:
            lcplot = lcplot.opts(xlim=x_range)

        return lcplot

    @param.depends("update_plot", "cmin", "cmax", "colormap")
    def plot_waterfall(self, x_range=None):
        ds = xr.Dataset(
            {"data": (["t", "channel"], self.stokesI)},
            coords={"channel": self.channels, "t": self.dt},
        )
        vmin, vmax = np.nanpercentile(self.stokesI, (self.cmin * 100, self.cmax * 100))
        clim = (vmin, vmax)

        waterfallplot = (
            ds.hvplot(
                colorbar=True,
                clim=clim,
                cmap=self.colormap,
                rasterize=True,
                dynamic=False,
            )
            .redim(x="t")
            .opts(
                width=self.width,
                height=int(self.width / 1.5),
                # Ensure the colorbar width doesn't change by
                # keeping the numbers on the scale fixed
                colorbar_opts={"formatter": PrintfTickFormatter(format="%.2f")},
            )
        )
        if x_range:
            waterfallplot = waterfallplot.opts(
                xlim=x_range,
                apply_ranges=False,
            )

        return waterfallplot

    @param.depends("update_plot")
    def plot_bkg(self):
        bkg_lc_mean = hv.Curve(
            (self.bkg["mean"], self.channels), "mean", "channel"
        ).opts(width=200, height=int(self.width / 1.5))
        bkg_lc_std = hv.Curve((self.bkg["std"], self.channels), "std", "channel").opts(
            width=200, height=int(self.width / 1.5)
        )

        return pn.Row(bkg_lc_mean, bkg_lc_std)

    def panel(self):
        dm_input = pn.Param(
            self.param.dm, widgets={"dm": {"type": pn.widgets.FloatInput, "width": 150}}
        )[0]
        dm_zoom = pn.Param(
            self.param.dm_zoom,
            widgets={"dm_zoom": {"type": pn.widgets.Select, "width": 150}},
        )[0]
        cmin = pn.Param(self.param.cmin, widgets={"cmin": pn.widgets.FloatInput})[0]
        cmax = pn.Param(self.param.cmax, widgets={"cmax": pn.widgets.FloatInput})[0]
        badchanlist = pn.Param(
            self.param.badchanlist,
            widgets={
                "badchanlist": {
                    "type": pn.widgets.TextInput,
                    "placeholder": "comma-separated integers",
                }
            },
        )[0]
        datarange = pn.Param(
            self.param.datarange, widgets={"datarange": pn.widgets.RangeSlider}
        )[0]
        datarange.width = self.width
        datarange.step = 0.01

        bkgplots = pn.Card(
            self.plot_bkg,
            header=pn.pane.HTML(
                '<h4 style="margin: 0; padding: 0; text-align: left">Background pre<br>bandpass-correction</h4>'
            ),
            collapsed=True,
        )
        waterfallplot = hv.DynamicMap(self.plot_waterfall, streams=[self.range_stream])
        self.tap.source = waterfallplot
        image_plot = pn.Column(waterfallplot, datarange)

        lcplot = hv.DynamicMap(self.plot_lc, streams=[self.range_stream])
        plots = pn.Column(lcplot, pn.Row(bkgplots, image_plot))
        trunclc = pn.Param(
            self.param.trunclc,
            widgets={"trunclc": {"type": pn.widgets.FloatInput, "width": 100}},
        )[0]
        lcsettings = pn.Card(
            pn.Row(
                self.param.loglc,
                trunclc,
            ),
            title="Light curve settings",
            collapsed=True,
        )
        colorsettings = pn.Card(
            pn.Column(
                self.param.logimage,
                self.param.colormap,
                pn.Row(cmin, cmax),
            ),
            title="Colormap settings",
            collapsed=True,
        )

        badchannels = pn.Card(
            badchanlist,
            title="Bad channels",
            collapsed=True,
        )
        dmsettings = pn.Card(
            pn.Row(
                self.dm_slider,
                dm_zoom,
                dm_input,
            ),
            title="Dispersion measure",
            collapsed=False,
        )
        layout = pn.Row(
            pn.Column(
                plots,
            ),
            pn.Column(
                lcsettings,
                colorsettings,
                badchannels,
                dmsettings,
            ),
        )
        return layout


def main(filename):
    with openfile(filename) as candidate:
        plot = CandidatePlot(candidate, invert=True, width=1200)
        layout = plot.panel()

        pn.serve(
            layout,
            title="FRB Candidate - interactive DM",
            port=5006,
            show=False,
            autoreload=True,  # This is equivalent to --dev
            dev=True,  # Enable development mode (includes autoreload + debug features)
            num_procs=1,  # Use single process (better for debugging)
        )


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("file", help="Filterbank file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.file)
