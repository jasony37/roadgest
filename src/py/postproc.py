import argparse
import proc.fd


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('fnames', help="Path of CSV(s) of estimator output", nargs='+')
    args = args_parser.parse_args()
    for fname in args.fnames:
        proc.fd.fit_fd(fname)

if __name__ == "__main__":
    main()