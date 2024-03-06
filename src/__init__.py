import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--timer_name", help="timer name")
parser.add_argument("-p", "--project", help="project name")
parser.add_argument('-s', '--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('-d', '--seed', type=int, default=72, help='Random seed.')
parser.add_argument('-e', '--epochs', type=int, default=10000, help='Number of epochs to train.')

args = parser.parse_args()
print(args.timer_name)
print(args.sparse)
print(args.seed)
print(args.epochs)
