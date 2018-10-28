import os
import argparse
import vis.core
import proc.readdata


def args_setup():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('datadir', help="Path to cabspottingdata")
    return args_parser.parse_args()


def main():
    args = args_setup()
    cab_traces_file = os.path.join(args.datadir, "cab_traces.pickle")
    cab_reader = proc.readdata.CabReader(args.datadir)
    cab_reader.read_cablist()
    if os.path.isfile(cab_traces_file):
        cab_reader.cabtraces_from_save(cab_traces_file)
    else:
        cab_reader.read_cabtraces()
        cab_reader.cab_traces.to_pickle(cab_traces_file)
    vis.core.plot_cabs_in_time(cab_reader.cab_traces, 1212991838, 1212991900)


if __name__ == "__main__":
    main()
