import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd

from moran import mixed_moran_stats, mixed_moran_abs_time_sys, mixed_moran_fix_prob_sys, mixed_moran_cond_fix_time_sys, vec_2_idx
from utils import istarmap
import tqdm
from multiprocessing import Pool

def work(N, G_name, G, r, p_Bd, trials, initial_mutant_location):
  # stats_df = mixed_moran_stats(G, r=r, p_Bd=p_Bd, n_trials=trials, initial_mutant_location=initial_mutant_location)
  # fp = stats_df['fixed'].sum() / trials
  # ft = stats_df[stats_df['fixed'] == True]['steps'].mean()
  # if G_name == 'complete' and r == 0.99 and p_Bd == 0:
  #   print(f'{N=} {G_name}, {r=}, {p_Bd=}, {fp=:.3f}, {ft=:.3f}')
  state = [0] * N
  state[initial_mutant_location] = 1
  state_idx = vec_2_idx(state) 
  fp = mixed_moran_fix_prob_sys(G, r=r, p_Bd=p_Bd)[state_idx]
  at = mixed_moran_abs_time_sys(G, r=r, p_Bd=p_Bd)[state_idx]
  ft = mixed_moran_cond_fix_time_sys(G, r=r, p_Bd=p_Bd)[state_idx]
  return (N, G_name, r, p_Bd, state_idx, fp, ft, at)

if __name__ == '__main__':
  N = 10
  TRIALS = int(1e4)

  Gs = [
    ('complete', nx.complete_graph(N), 0),
    ('cycle', nx.cycle_graph(N), 0),
    ('star', nx.star_graph(N-1), 0),
    ('star', nx.star_graph(N-1), 1),
    ('path', nx.path_graph(N), 0),
    ('path', nx.path_graph(N), 1),
  ]
  assert all(len(G) == N for _, G, _ in Gs)

  data = []
  with Pool() as pool:
    for datum in tqdm.tqdm(
      pool.istarmap(
        work,
        [
          (N, G_name, G, r, p_Bd, TRIALS, initial_mutant_location)
          for G_name, G, initial_mutant_location in Gs
          for r in np.linspace(.5, 1.5, 10+1)
          for p_Bd in np.linspace(0, 1, 10+1)
        ],
      ),
      total=len(Gs) * 11 * 11,
    ):
      data.append(datum)

  df = pd.DataFrame(data, columns=['N', 'graph', 'r', 'p_Bd', 'initial_mutant_location', 'fix_prob', 'fix_time', 'abs_time'])
  df.to_csv(f'./data/moran-stats-{N}-results.csv', index=False)
