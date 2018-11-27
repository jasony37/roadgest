import os
import argparse
import numpy as np

import vis.core
import proc.cabdata
import proc.road


def args_setup():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('datadir', help="Path to cabspottingdata")
    args_parser.add_argument('road', help="Path to road section CSV")
    return args_parser.parse_args()


def main():
    args = args_setup()
    cab_data = proc.cabdata.CabData(args.datadir)
    road_section = proc.road.RoadSection(args.road)
    cab_data.calc_xy(*road_section.center)
    cab_data.calc_deltas()
    cab_data.assign_road_segments(road_section, 20.0, np.deg2rad(30.0), (1211298000, 1211301600))
    vis.core.plot_cabs_in_time(cab_data.cab_traces, (1211298000, 1211301600), road_section.extents)
    #vis.core.plot_timestamps(cab_data.cab_traces, 1212991838, 1212995438)


if __name__ == "__main__":
    main()
