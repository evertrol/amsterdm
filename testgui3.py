from bokeh.models import PrintfTickFormatter
import holoviews as hv
from holoviews import streams
import hvplot
import hvplot.xarray  # noqa: F401
import numpy as np
import panel as pn
import param
import xarray as xr


hv.extension("bokeh")
pn.extension()



class ExamplePlot(param.Parameterized):
    offset = param.Number(default=0.0, bounds=(0.0, 2.0), step=0.1, label="Offset")
    start = param.Number(default=0)
    end = param.Number(default=10)
    reset = param.Action(lambda x: x.param.trigger('reset'))
    action = param.Action(lambda x: x.param.trigger('action'))
    update_plot = param.Number(default=0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.range_stream = hv.streams.RangeX()
        self._init_data()

        self.param.watch(
            self._update_data,
            [
                "offset",
                "start",
                "end",
            ],
        )

        #self.param.watch(self.reset_range, "reset")

    @param.depends('action')
    def get_number(self):
        print('getting number')

    def _init_data(self):
        self._calc_data()

    def _calc_data(self):
        print('calc data')
        self.x = np.linspace(self.start, self.end)
        self.y = np.linspace(0, 5)
        X, Y = np.meshgrid(self.x, self.y)
        self.data = np.sin(X+self.offset) * np.cos(Y)

    def _update_data(self, event):
        self._calc_data()
        self.update_plot += 1

    @param.depends("reset")
    def reset_range(self):
        self.start = 0
        self.end = 10
        print('resetting range')
        #self._update_data(

    @param.depends("update_plot")
    def plot(self, x_range=None):
        ds = xr.Dataset(
            {"data": (["y", "x"], self.data.T)},
            coords={"x": self.x, "y": self.y},
        )

        plot = (
            ds.hvplot(
                colorbar=True,
                dynamic=False,
            )
#            .redim(x="x")
        )

        if x_range:
            plot = plot.opts(
                xlim=x_range,
            )

        return plot

    def view(self):
        dmap = hv.DynamicMap(
            self.plot,
            streams=[self.range_stream]
        )

        controls = pn.Param(
            self,
            parameters=["start", "end"],
            widgets={
                "start": pn.widgets.FloatInput,
                "end": pn.widgets.FloatInput,
            },
            show_name=False,
        )

        return pn.Column(
            pn.pane.Markdown("## Image zoom persistence demo"),
            controls,
            self.param.reset,
            self.param.action,
            self.param.offset,
            dmap,
        )


def main():
    plot = ExamplePlot()
    layout = plot.view()

    pn.serve(
        layout,
        title="Example plot",
        port=5006,
        show=False,
        autoreload=True,
        dev=True,
    )

if __name__ == "__main__":
    main()
