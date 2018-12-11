import argparse
import proc.fd


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--training', help="Path of CSV(s) of estimator output for training", nargs='+')
    args_parser.add_argument('--test', help="Path of CSV(s) of estimator output for testing", nargs='+')
    args = args_parser.parse_args()
    proc.fd.fit_fd(args.training)

if __name__ == "__main__":
    main()