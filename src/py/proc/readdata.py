import pandas as pd
import warnings


class CabListReader(object):
    def __init__(self, fname):
        self._fname = fname

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

    def read_data(self):
        cab_list = []
        with open(self._fname, 'r') as fin:
            for line in fin:
                cab_data = self._proc_line(line)
                if cab_data is not None:
                    cab_list.append(cab_data)
