#!/usr/bin/env python3.11
import matplotlib.pyplot as plt
import moran
import networkx as nx
import numpy as np
import pandas as pd
import pprint
import random
import seaborn as sns
import tqdm

from dataclasses import dataclass
from collections import defaultdict
from multiprocessing import Pool
from typing import *

Configuration = Dict[int, moran.Type]
HashedConfiguration = Tuple[int, ...]


@dataclass
class Step:
  before: Configuration
  after: Configuration

def random_start_one_step(
  G: nx.Graph,
  r: float,
  p_Bd: float,
) -> Step:
  loc_to_type = {
    node: moran.Type.RESIDENT if random.uniform(0, 1) < 0.5 else moran.Type.MUTANT
    for node in G.nodes()
  }
  before_loc_to_type = loc_to_type.copy()

  fitness = lambda t: r if t == moran.Type.MUTANT else 1

  # Step.
  if random.uniform(0, 1) < p_Bd:
    # Bd step
    birther = random.choices(population=list(G.nodes()), weights=[fitness(loc_to_type[node]) for node in G.nodes()])[0]
    dier = random.choice(list(G.neighbors(birther)))
  else:
    # dB step
    dier = random.choice(list(G.nodes()))
    dier_neighbors = list(G.neighbors(dier))
    birther = random.choices(population=dier_neighbors, weights=[fitness(loc_to_type[node]) for node in dier_neighbors])[0]

  loc_to_type[dier] = loc_to_type[birther]
  
  return Step(
    before=before_loc_to_type,
    after=loc_to_type,
  )


PotentialFn = Callable[[Configuration, nx.Graph, float], float] 

def simple(configuration: Configuration, G: nx.Graph, p_Bd: float) -> float:
  return sum(1 for u, t in configuration.items() if t == moran.Type.MUTANT) / len(G)

def Bd(configuration: Configuration, G: nx.Graph, p_Bd: float) -> float:
  return (
    sum(1/G.degree(u) for u, t in configuration.items() if t == moran.Type.MUTANT)
    / sum(1/G.degree(u) for u, t in configuration.items())
  )

def dB(configuration: Configuration, G: nx.Graph, p_Bd: float) -> float:
  return (
    sum(G.degree(u) for u, t in configuration.items() if t == moran.Type.MUTANT)
    / sum(G.degree(u) for u, t in configuration.items())
  )

def mixed(configuration: Configuration, G: nx.Graph, p_Bd: float) -> float:
  return p_Bd * Bd(configuration, G, p_Bd) + (1 - p_Bd) * dB(configuration, G, p_Bd)

def exponent(configuration: Configuration, G: nx.Graph, p_Bd: float) -> float:
  return (
    sum(G.degree(u) ** (1-2*p_Bd) for u, t in configuration.items() if t == moran.Type.MUTANT)
    / sum(G.degree(u) ** (1-2*p_Bd) for u, t in configuration.items())
  )

def stupid(configuration: Configuration, G: nx.Graph, p_Bd: float) -> float:
  return 1

POTENTIAL_FNS: List[PotentialFn] = [
  stupid,
  simple,
  Bd,
  dB,
  mixed,
  exponent,
]

def hash_configuration(configuration: Configuration) -> HashedConfiguration:
  return tuple(t.value for u, t in sorted(configuration.items(), key=lambda x: x[0]))

if __name__ == '__main__':
  N = 10
  TRIALS = (1 << N) * 5
  G = nx.star_graph(N-1) 
  r = 1
  p_Bd = 1
  with Pool() as pool:
    simulation_results = pool.starmap(
      random_start_one_step, [
      (G, r, p_Bd)
      for _ in range(TRIALS)
    ], chunksize=TRIALS // pool._processes)


  potential_fn_to_values: Dict[PotentialFn, DefaultDict[HashedConfiguration, List[float]]] = {}
  for potential_fn in POTENTIAL_FNS:
    potential_values: DefaultDict[HashedConfiguration, float] = defaultdict(list)

    for step in simulation_results:
      potential_before = potential_fn(step.before, G, p_Bd)
      potential_after = potential_fn(step.after, G, p_Bd)
      potential_values[hash_configuration(step.before)].append(potential_after-potential_before)

    potential_fn_to_values[potential_fn] = potential_values

  df = pd.DataFrame(
    [
      {
        'potential_fn': potential_fn.__name__,
        'configuration': ''.join(map(str, configuration)),
        'potential': value,
      }
      for potential_fn, potential_values in potential_fn_to_values.items()
      for configuration, values in potential_values.items()
      for value in values
    ]
  )
  averages_df = (
    df
    .groupby(['potential_fn', 'configuration'])
    .mean()
    .reset_index()
    .sort_values(['potential_fn', 'potential'], ascending=False)
    .reset_index()
    .drop(columns=['index'])
  )
  print(averages_df)
  fg = sns.FacetGrid(data=averages_df, col='potential_fn', sharey=True, sharex=False, height=2.5, aspect=2, col_wrap=3)
  fg.map_dataframe(
    sns.barplot,
    x='configuration',
    y='potential',
    dodge=False,
    width=1.0,
    errorbar=None,
  )
  print(averages_df.drop(columns=['configuration']).groupby('potential_fn').mean().sort_values('potential', ascending=False))
  plt.show()