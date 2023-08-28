from inn_model import *
import config as c

import argparse

parser = argparse.ArgumentParser(description="Define test")
parser.add_argument("condition", type=int, nargs=1)

args = parser.parse_args()

cond = torch.Tensor([args.condition[0], ])

def main():
    visual_test(cond)

if __name__ == "__main__":
    main()
