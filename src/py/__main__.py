import os
import vis.core
import proc.readdata


def main():
    cabs_file = r"C:\Users\Jason\Desktop\BDownloads\cabspottingdata"
    cab_traces_file = os.path.join(cabs_file, "cab_traces.pickle")
    cab_reader = proc.readdata.CabReader(cabs_file)
    cab_reader.read_cablist()
    if os.path.isfile(cab_traces_file):
        cab_reader.cabtraces_from_save(cab_traces_file)
    else:
        cab_reader.read_cabtraces()
        cab_reader.cab_traces.to_pickle(cab_traces_file)
    vis.core.plot_cabs_in_time(cab_reader.cab_traces, 1212991838, 1212991900)


if __name__ == "__main__":
    main()
