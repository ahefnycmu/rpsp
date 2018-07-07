from argparse import ArgumentParser
from multiprocessing import Pool
from os import system
import subprocess

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--cmd', type=str, required=True)
    parser.add_argument('--key', type=str, default='%param')
    parser.add_argument('--vals', type=str, default='')
    parser.add_argument('--n', type=int, required=True)
    args = parser.parse_args()

    def f(s):
        cmd = args.cmd.replace(args.key, s)
        #system(cmd)
        print cmd
        p = subprocess.call(cmd, shell=True)

    p = Pool(args.n)

    vals = args.vals.split()
    p.map(f, vals)
