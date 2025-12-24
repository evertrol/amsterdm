from pathlib import Path
import sys

import amsterdm
import amsterdm.plots


def main(path):
    with amsterdm.openfile(path) as candidate:
        print(candidate.data.shape)

        dm = 219
        badchannels = [1,9,76,79,80,81,82,83,98,99,100,103,111,112,113,123]
        ax = amsterdm.plots.waterfall(candidate, 230, badchannels, datarange=[0, 0.4])

        ax.set_title('DM of some burst')

        outfile = path.with_suffix('.png')
        outfile = outfile.with_stem(path.stem + '-waterfall')
        print(outfile)
        ax.figure.savefig(outfile)


        dm = 150
        ax = amsterdm.plots.bowtie(candidate, (dm-100, dm+100), badchannels, datarange=[0.2, 0.8], ndm=20)

        outfile = path.with_suffix('.png')
        outfile = outfile.with_stem(path.stem + '-bowtie')
        print(outfile)
        ax.figure.savefig(outfile)





def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("file", help="Filterbank file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.file))
