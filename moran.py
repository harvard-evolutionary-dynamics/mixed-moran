from __future__ import annotations
import networkx as nx
import random
import enum

import numpy as np

from itertools import product

class Type(enum.Enum):
  RESIDENT = 0
  MUTANT = 1

def mixed_moran_abs_time(G: nx.Graph, *, r: float, p_Bd: float):
  N = len(G)
  assert N > 0

  loc_to_type = {
    node: Type.RESIDENT
    for node in G.nodes()
  }
  initial_mutant_location = random.choice(list(G.nodes()))
  loc_to_type[initial_mutant_location] = Type.MUTANT
  num_mutants = 1

  steps = 0
  fitness = lambda t: r if t == Type.MUTANT else 1
  while 0 < num_mutants < N:
    F = N + num_mutants * (r-1)
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

    num_mutants += loc_to_type[birther].value - loc_to_type[dier].value
    loc_to_type[dier] = loc_to_type[birther]
    steps += 1

  return (steps, loc_to_type[0])

def vec_2_idx(S):
  return int(''.join(map(str, S)), base=2)

def mixed_moran_abs_time_sys(G: nx.Graph, *, r: float, p_Bd: float):
  b = np.ones(2**len(G))
  b[0] = b[-1] = 0

  return mixed_moran_master_sys(G, r=r, p_Bd=p_Bd, b=b)

def mixed_moran_fix_prob_sys(G: nx.Graph, *, r: float, p_Bd: float):
  b = np.zeros(2**len(G))
  b[0] = 0
  b[-1] = 1
  return mixed_moran_master_sys(G, r=r, p_Bd=p_Bd, b=b)

def mixed_moran_master_sys(G: nx.Graph, *, r: float, p_Bd: float, b: np.array):
  N = len(G)
  A = np.zeros((2**N, 2**N))

  non_zero_idxs = lambda S: (idx for idx, e in enumerate(S) if e != 0)
  zero_idxs = lambda S: (idx for idx, e in enumerate(S) if e == 0)
  W = lambda S: sum((1 if e == 0 else r) for e in S)
  F = lambda u, S: sum((1 if S[v] == 0 else r) for _, v in G.edges(u))


  for row, S in enumerate(product([0,1], repeat=N)):
    if row == 0:
      A[row, 0] = 1
      continue
    if row == 2**N-1:
      A[row, -1] = 1
      continue
    
    S_idx = vec_2_idx(S)

    # Bd.
    A[row, S_idx] += p_Bd * (
      1
      - sum(
        r/W(S) * 1/G.degree(u)
        for u in non_zero_idxs(S)
        for v in non_zero_idxs(S)
        if G.has_edge(u, v)
      )
      - sum(
        1/W(S) * 1/G.degree(u)
        for u in zero_idxs(S)
        for v in zero_idxs(S)
        if G.has_edge(u, v)
      )
    )
    for u in non_zero_idxs(S):
      for v in zero_idxs(S):
        if G.has_edge(u, v):
          Sp = list(S)
          Sp[v] = 1
          A[row, vec_2_idx(Sp)] += p_Bd*(-r/W(S) * 1/G.degree(u))
    for u in zero_idxs(S):
      for v in non_zero_idxs(S):
        if G.has_edge(u, v):
          Sp = list(S)
          Sp[v] = 0
          A[row, vec_2_idx(Sp)] += p_Bd*(-1/W(S) * 1/G.degree(u))


    # dB.
    A[row, S_idx] += (1-p_Bd) * (
      1
      - sum(
        1/N * r/F(u, S)
        for u in non_zero_idxs(S)
        for v in non_zero_idxs(S)
        if G.has_edge(u, v)
      )
      - sum(
        1/N * 1/F(u, S)
        for u in zero_idxs(S)
        for v in zero_idxs(S)
        if G.has_edge(u, v)
      )
    )
    for u in non_zero_idxs(S):
      for v in zero_idxs(S):
        if G.has_edge(u, v):
          Sp = list(S)
          Sp[u] = 0
          A[row, vec_2_idx(Sp)] += (1-p_Bd)*(-1/N * 1/F(u, S))
    for u in zero_idxs(S):
      for v in non_zero_idxs(S):
        if G.has_edge(u, v):
          Sp = list(S)
          Sp[u] = 1
          A[row, vec_2_idx(Sp)] += (1-p_Bd)*(-1/N * r/F(u, S))

  print(A)
  print(b)
  t = np.linalg.solve(A, b)
  return t