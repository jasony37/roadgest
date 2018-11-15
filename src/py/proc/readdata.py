import os
import pandas as pd
import warnings


class CabData(object):
    def __init__(self, dirname):
        self._dirname = dirname
        self._fname = os.path.join(dirname, "_cabs.txt")
        self.cab_list = None
        self.cab_traces = None
        cab_traces_file = os.path.join(dirname, "cab_traces.pickle")
        self.read_cablist()
        if os.path.isfile(cab_traces_file):
            self.cabtraces_from_save(cab_traces_file)
        else:
            self.read_cabtraces()
            self.cab_traces.to_pickle(cab_traces_file)

    def _tag_value(self, line, tag_name):
        try:
            search_str = '{}="'.format(tag_name)
            start_idx = line.index(search_str) + len(search_str)
            end_idx = line.index('"', start_idx)
            return line[start_idx:end_idx]
        except ValueError as valerr:
            print(valerr)

    def _proc_line(self, line):
        line_s = line.rstrip()
        if line_s.startswith('<') and line_s.endswith('/>'):
            cab_id = self._tag_value(line_s, "cab id")
            updates = int(self._tag_value(line_s, "updates"))
            cab = {'id': cab_id, 'updates': updates}
            return cab
        else:
            warnings.warn("Line does not look like a tag. Ignored!")

    def read_cablist(self):
        cab_list = []
        with open(self._fname, 'r') as fin:
            for line in fin:
                cab_data = self._proc_line(line)
                if cab_data is not None:
                    cab_list.append(cab_data)
        self.cab_list = pd.DataFrame(cab_list)

    def cab_id_to_fname(self, cab_id):
        return os.path.join(self._dirname, "new_{}.txt".format(cab_id))

    def read_cabtraces(self):
        assert(self.cab_list is not None)
        id_col = self.cab_list['id'].apply(self.cab_id_to_fname)
        flist = self.cab_list.assign(fname=id_col)
        cab_data_list = []
        for cabf in flist.itertuples():
            cab_data = pd.read_csv(cabf.fname, delim_whitespace=True,
                                   names=['lat', 'long', 'occupancy', 'time'])
            cab_data_list.append(cab_data.assign(cab_id=cabf.id))
        self.cab_traces = pd.concat(cab_data_list, ignore_index=True)

    def cabtraces_from_save(self, fname):
        self.cab_traces = pd.read_pickle(fname)


class RoadSection(object):
    def __init__(self, fname):
        self.section = pd.read_csv(fname)