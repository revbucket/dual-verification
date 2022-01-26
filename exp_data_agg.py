import pickle
import argparse
from pathlib import Path
import pandas as pd
import itertools as it
import re

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
args = parser.parse_args()
root = args.dir

assert root.exists()

algs = set()
props = ["bound", "time"]
data = {}

for f in root.glob("[0-9]*.pkl"):
    i = int(re.match("[0-9]+", f.stem).group())
    run_data = pickle.load(open(f, "rb"))
    data.setdefault(i, {}).update(run_data)
    algs.update(run_data.keys())

print(sorted(algs))

df = pd.DataFrame(columns=list(it.product(algs, props)))

for i in sorted(data.keys()):
    df.loc[i] = {
        (alg, col): t[j] for alg, t in data[i].items() for j, col in enumerate(props)
    }

print(df)

df.to_hdf(root.parent / f"{root.stem}.hdf5", key="data")
