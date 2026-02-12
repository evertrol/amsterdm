#! /usr/bin/env python

"""Run examples/simulate.py to create an input file for this example"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import amsterdm
import amsterdm.plot as dmplot


logger = logging.getLogger("amsterdm")


def setup_logger(loglevel):
    fmt = "%(asctime)s  [%(levelname)-5s] - %(module)s.%(funcName)s():%(lineno)d: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(loglevel)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def main(path, dm, plots, background, badchannels=None, loglevel=logging.INFO):
    setup_logger(loglevel)
    logger.info("Reading file %s", path)
    with amsterdm.openfile(path) as burst:
        logger.info("Done reading file")
        pngfile = path.with_suffix(".png")
        if badchannels is None:
            badchannels = []

        plt.figure(figsize=(12, 6))
        ax = plt.subplot(1, 2, 1)
        nsamples = burst.data.shape[0]
        sections = [(62334, 62633), (65462, 65782)]
        sections = [
            (section[0] / nsamples, section[1] / nsamples) for section in sections
        ]

        dms = np.linspace(dm - 1.15, dm + 1.15, 50)
        dmplot.signal2noise(
            burst,
            dms,
            badchannels=badchannels,
            backgroundrange=background,
            peak=True,
            peak_interval=sections[0],
            fit=True,
            ax=ax,
        )

        ax = plt.subplot(1, 2, 2)
        dmplot.signal2noise(
            burst,
            dms,
            badchannels=badchannels,
            backgroundrange=background,
            peak=True,
            peak_interval=sections[1],
            fit=True,
            ax=ax,
        )

        plt.savefig(pngfile.with_stem(path.stem + "-s2n-multipeaks"))


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("file", help="Filterbank file")
    parser.add_argument("--dm", type=float, default=219.356)
    parser.add_argument(
        "--plots",
        choices=(
            "waterfall",
            "dynspec",
            "bowtie",
            "lightcurve",
            "lc",
            "background",
            "bg",
            "ratio",
            "s2n",
            "all",
            "grid",
        ),
        default="all",
        nargs="+",
    )
    parser.add_argument(
        "--badchannels", type=int, nargs="+", help="list of channel numbers"
    )
    parser.add_argument(
        "--back",
        nargs=2,
        type=float,
        action="append",
        help="Set of start and end (time) fractions for the background estimate",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.file), args.dm, args.plots, args.back, badchannels=args.badchannels)
