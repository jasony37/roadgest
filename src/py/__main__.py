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
    cab_data = proc.readdata.CabData(args.datadir)
    vis.core.plot_cabs_in_time(cab_data.cab_traces, 1212991838, 1212995438)
    #vis.core.plot_timestamps(cab_data.cab_traces, 1212991838, 1212995438)


if __name__ == "__main__":
    main()
