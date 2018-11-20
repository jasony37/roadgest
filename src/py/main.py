import os
import argparse
import vis.core
import proc.readdata


def args_setup():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('datadir', help="Path to cabspottingdata")
    args_parser.add_argument('road', help="Path to road section CSV")
    return args_parser.parse_args()


def main():
    args = args_setup()
    cab_data = proc.readdata.CabData(args.datadir)
    road_section = proc.readdata.RoadSection(args.road)
    cab_data.calc_xy(*road_section.center)
    cab_data.check_on_section(road_section, 20.0, (1211295600, 1211298000))
    vis.core.plot_cabs_in_time(cab_data.cab_traces, 1211295600, 1211298000)
    #vis.core.plot_timestamps(cab_data.cab_traces, 1212991838, 1212995438)


if __name__ == "__main__":
    main()
