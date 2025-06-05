#!/usr/bin/env python3.11
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from collections import deque
from multiprocessing import Pool
from pathlib import Path
from utils import yield_all_graph6, istarmap
from moran import (
  mixed_moran_fix_prob_sys,
  mixed_moran_abs_time_sys,
  mixed_moran_cond_fix_time_sys,
  vec_2_idx,
)


def process(G: nx.Graph, r: float, p_Bd: float):
  # print(f'Processing graph {G.number_of_nodes()} with r={r} and p_Bd={p_Bd}')
  N = len(G)
  fix_time = mixed_moran_cond_fix_time_sys(G, r=r, p_Bd=p_Bd)
  fix_prob = mixed_moran_fix_prob_sys(G, r=r, p_Bd=p_Bd)

  # Bd temperature.
  T = lambda v: sum(1/G.degree(u) for u in G.neighbors(v))/N

  S = deque([1] + [0]*(N-1))
  values_fix_time_uniform = [None]*N
  values_fix_prob_uniform = [None]*N
  values_fix_time_temp = 0
  values_fix_prob_temp = 0
  for i in range(N):
    values_fix_time_uniform[i] = fix_time[vec_2_idx(S)]
    values_fix_prob_uniform[i] = fix_prob[vec_2_idx(S)]
    values_fix_time_temp += fix_time[vec_2_idx(S)] * ((p_Bd)*T(i) + (1-p_Bd)/N)
    values_fix_prob_temp += fix_prob[vec_2_idx(S)] * ((p_Bd)*T(i) + (1-p_Bd)/N)
    S.rotate(1) 

  return {
    'N': N,
    'graph': nx.graph6.to_graph6_bytes(G, header=False).hex(),
    'p_Bd': p_Bd,
    'r': r,
    'fix_time_uniform': np.mean(values_fix_time_uniform),
    'fix_prob_uniform': np.mean(values_fix_prob_uniform),
    'fix_time_temp': values_fix_time_temp,
    'fix_prob_temp': values_fix_prob_temp,
  } | {
    f'fix_time_uniform_{i+1}': v
    for i, v in enumerate(values_fix_time_uniform)
  } | {
    f'fix_prob_uniform_{i+1}': v
    for i, v in enumerate(values_fix_prob_uniform)
  }

def main():
  # Constants.
  N = 7
  Ls = np.linspace(0, 1, 10+1)
  Rs = np.linspace(.25, 1.75, 10+1)

  path = Path(f'./data/connected-n{N}.g6')
  with path.open("rb") as f:
    num_graphs = sum(1 for _ in f)

  data = []
  with Pool() as pool:
    for result in tqdm.tqdm(pool.istarmap(
        process,
        (
          (G, r, p_Bd)
          for G in yield_all_graph6(Path(f'./data/connected-n{N}.g6'))
          for r in Rs
          for p_Bd in Ls
        )
      ),
      total=num_graphs*len(Rs)*len(Ls),
    ):
      data.append(result)

  df = pd.DataFrame(data)
  df.to_csv(f'./data/connected-n{N}-results.csv')

if __name__ == '__main__':
  main()