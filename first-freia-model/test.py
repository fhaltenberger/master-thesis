from inn_model import *
import config as c

import argparse

parser = argparse.ArgumentParser(description="Define test")
parser.add_argument("test_type", nargs="*")

args = parser.parse_args()

test_type = args.test_type   

def main():
    if len(test_type) == 2:
        if test_type[0] == "latent": 
            plot_latents(dim=int(test_type[1]))
        else: raise ValueError("Argument has to be 'latent', 'latent <integer>' or '<integer>'.")
    if len(test_type) == 1:
        visual_test(int(test_type[0]))
    else: raise ValueError("Argument has to be 'latent', 'latent <integer>' or '<integer>'.") 

if __name__ == "__main__":
    main()
