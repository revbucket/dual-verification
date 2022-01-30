# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import subprocess
import re

# helper to melt the first level of a MultiIndex
def melt1(df, name="alg"):
    return (df.unstack()
        .unstack(level=1)
        .reset_index(level=1, drop=True)
        .rename_axis(name)
        .reset_index())

def diff1(df, alg_ref, drop=True):
    df = df.reindex(sorted(df.columns), axis=1)
    cols = df.columns
    if drop:
        cols = cols.drop(alg_ref, level=0)
    df2 = pd.DataFrame(columns=cols)
    for (alg, prop) in df2.columns:
        df2[alg, prop] = df[alg, prop] - df[alg_ref, prop] 
    return df2

def custom_round(mu, sigma, n, shift=0):
    digits = int(np.ceil(-np.log10(12.44/np.sqrt(n)))) + shift
    out = round(mu, digits), round(sigma, digits)
    if np.isclose(out, 0.0, atol=1e-9).any() and shift < 4:
        return custom_round(mu, sigma, n, shift+1)
    return out

def drop_outliers(series, r=5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series[~((series < (Q1 - r * IQR)) |(series > (Q3 + r * IQR)))]

def drop_outliers2(df, r=5):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - r * IQR)) |(df > (Q3 + r * IQR))).any(axis=1)]

# %config InlineBackend.print_figure_kwargs={'facecolor' : "w"}

sns.set_theme()

# mpl.use("pgf")
sns.set(font_scale=0.7)

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 1,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'legend.fontsize': 7,
    'legend.title_fontsize': 7,
    'legend.handlelength' : 2
})

def savefig(name):
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
    plt.savefig(f"figures/{name}.eps", **kwargs)
    plt.savefig(f"figures/{name}.pgf", **kwargs)
    subprocess.run(['epstopdf', name + ".eps"], cwd="figures")


# %%
# colors = sns.color_palette("husl", 8)
colors = sns.color_palette("tab10")
colors

# %%
# algs = ["decomp_2d", "decomp_mip", "optprox", "explp", "anderson", "lp"]
props = ["bound", "time"]
# files = ["mnist_ffnet", "mnist_deep", "mnist_wide"]

palette = {k: colors[i] for k, i in {
    "optprox": 9,
    "optprox_all": 9,
    "lp": 7,
    "lp_all": 7,
    "anderson": 8,
    "explp": 4,
    "explp256_all": 3,
    "explp512_all": 2,
    "decomp_2d": 0,
    "decomp_mip": 1,
    "decomp_mip_explp256": 0,
    "decomp_mip_explp512": 1,
    "decomp_mip_optprox": 6
}.items()}

name_subs = {
    "optprox": "BDD+",
    "optprox_all": "BDD+",
    "explp": "AS",
    "explp256_all": "AS256",
    "explp512_all": "AS512",
    "anderson": "Anderson",
    "lp": "LP",
    "lp_all": "LP",
    "decomp_2d": r"\textbf{ZD-2D}",
    "decomp_mip": r"\textbf{ZD-MIP}",
    "decomp_mip_optprox": r"\textbf{BDD+ → ZD}",
    "decomp_mip_explp256": r"\textbf{AS256 → ZD}",
    "decomp_mip_explp512": r"\textbf{AS512 → ZD}"
}

experiment_subs = {
    "mnist_ffnet": "MNIST, FFNet",
    "mnist_wide": "MNIST, Wide",
    "mnist_deep": "MNIST, Deep",
    "mnist_ffnet_all": "MNIST, FFNet, All Layers",
    "mnist_wide_all": "MNIST, Wide, All Layers",
    "mnist_deep_all": "MNIST, Deep, All Layers",
    "cifar_sgd": "CIFAR, SGD",
}

def fix_legend(legend):
    legend.set_title("Algorithm")
    for t in legend.texts:
        t.set_text(name_subs.get(t.get_text(), t.get_text()))


# %%
mnist_ffnet = pd.read_hdf("exp_data/mnist_ffnet.hdf5")
mnist_ffnet

# %%
bounds = [
    (-50, 5), 
    (-50, 10), 
    (-100, 5), 
    (-40, 10),
    (-45, 7),
    (-75, 7),
    (-40, 15)
]
for (exp, name), (xmin, xmax) in zip(experiment_subs.items(), bounds):
    df = pd.read_hdf(f"exp_data/{exp}.hdf5")
    plt.figure(figsize=(3.5,2.0))
    meds = df.xs('bound', axis=1, level=1).median()
    ax = sns.ecdfplot(
        x="bound",
        hue="alg",
        hue_order=meds.sort_values().keys(),
        palette = palette,
        data=melt1(df),
        zorder=1
    )
    for i, l in enumerate(ax.lines):
        l.set_zorder(10 + len(ax.lines) - i)
    plt.xlim(xmin, xmax)
    plt.xlabel("Bound")
#     plt.title(f"CDF plot of bounds ({name})")
    fix_legend(ax.get_legend())
    savefig(f"{exp}_cdf")
    plt.show()

# %%
baseline = ["decomp_mip"]*3 + ["decomp_mip_explp256"]*3 + ["decomp_mip"]
bounds = [
    (-20, 1), 
    (-27, 0), 
    (-60, 3), 
    (-15, 2),
    (-15, 4),
    (-25, 4),
    (-20, 2)
]
for (exp, name), base, (xmin, xmax) in zip(experiment_subs.items(), baseline, bounds):
    df = pd.read_hdf(f"exp_data/{exp}.hdf5")
    plt.figure(figsize=(3.5,2.0))
    diffs = diff1(df, base) #.drop("decomp_2d", axis=1, level=0)
    means = diffs.xs('bound', axis=1, level=1).mean()
    ax = sns.kdeplot(
        x="bound",
        hue="alg",
        hue_order=means.sort_values().keys(),
        palette = palette,
        data=melt1(diffs),
        cut=1
    )
    ax.axvline([0], linewidth=1, color='grey', linestyle="--", label=name_subs[base])
    plt.xlabel(f"Bound Difference (other − reference)")
#     plt.title(f"{name_subs[base]} vs. Others ({name})")
    plt.tick_params(axis='y', left=False, labelleft=False)
    plt.xlim(xmin, xmax)
    fix_legend(ax.get_legend())
    savefig(f"{exp}_diff")
    plt.show()

