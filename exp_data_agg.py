import pickle
import argparse
from pathlib import Path
import pandas as pd
import itertools as it

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
args = parser.parse_args()
root = args.dir

assert root.exists()

algs = ["decomp_2d", "decomp_mip", "optprox", "explp", "anderson", "lp"]
props = ["bound", "time"]
df = pd.DataFrame(columns=list(it.product(algs, props)))

for f in sorted(root.glob("[0-9]*.pkl"), key=lambda f: int(f.stem)):
    i = int(f.stem)
    data = pickle.load(open(f, "rb"))
    df.loc[i] = {
        (alg, col): t[j] for alg, t in data.items() for j, col in enumerate(props)
    }

print(df)

df.to_hdf(root.parent / f"{root.stem}.hdf5", key="data")
