import argparse

# 创建解释器
parser = argparse.ArgumentParser()


parser.add_argument("--a",type=int, help="operator A")
parser.add_argument("--b",type=int, help="operator B")

args = parser.parse_args()

print(args)