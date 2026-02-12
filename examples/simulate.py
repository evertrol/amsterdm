from argparse import ArgumentParser

import astropy.io.fits as pyfits

import amsterdm


def run(overwrite: bool = False):
    peakfluxes = [5.5, 8]  # , 8, 4, 2]
    t0s = [0.1, 0.105]  # , 0.15, 0.18, 0.22]  # in seconds
    widths = [1e-4, 1e-4]  # * 4  # peak width in seconds

    dm = [123.35, 123.45]
    dmreffreq = 1400
    nsamples = 150000
    nchannels = 64
    time0 = 0
    timestep = 1.6e-6  # 64 microsec resolution
    freq0 = 1400
    dfreq = -2
    xy = [1, 0.5]
    # bandpass = np.sin(np.linspace(0, np.pi, nchannels))
    bandpass = 10  # constant bandpass of 100
    background = (5, 2)  # value and noise of background

    simdata, times, freqs = amsterdm.sim.simulate(
        peakfluxes,
        t0s,
        widths,
        dm,
        dmreffreq,
        nsamples,
        nchannels,
        freq0,
        dfreq,
        time0,
        timestep,
        xy=xy,
        bandpass=bandpass,
        background=background,
    )

    header = pyfits.Header()
    header["object"] = ("N/A", "source name")
    header["srcname"] = ("N/A", "source name")
    if isinstance(dm, (list, tuple)):
        dm = dm[0]
    header["coh_dm"] = (dm, "coherent dispersion measure")
    header["fchan1"] = (1400, "frequency of channel 1 [MHz]")
    header["foff"] = (dfreq, "frequency width of a channel")
    header["fanchor"] = ("mid", "anchor point on channel (top, bottom, mid)")
    header["badchan"] = ("", "comma-separated list of bad channels")
    header["telescop"] = ("N/A", "telescope name")
    header["ra"] = (0, "source right ascension")
    header["dec"] = (0, "source declination")
    header["tstart"] = (0, "time zero in MJD")
    header["tsamp"] = (timestep, "sampling time interval in seconds")
    header["observer"] = ("amsterdm", "observer")
    header["polchan"] = ("xx-yy", "polarization channel type")

    hdu = pyfits.PrimaryHDU(header=header, data=simdata)
    hdu.writeto("amsterdm-sim.fits", overwrite=overwrite)


def main():
    parser = ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
