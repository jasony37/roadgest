import argparse
import proc.fd


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--training', help="Path of CSV(s) of estimator output for training", nargs='+')
    args_parser.add_argument('--test', help="Path of CSV(s) of estimator output for testing", nargs='+')
    args = args_parser.parse_args()
    fd_fits = proc.fd.fit_fds(args.training)
    proc.fd.test_fd(args.test, fd_fits)

if __name__ == "__main__":
    main()