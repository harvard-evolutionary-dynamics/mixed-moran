import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd

from moran import mixed_moran_abs_time

if __name__ == '__main__':
  N = 8
  r = 1

  G = nx.complete_graph(N)

  data = [
    ('complete', N, r, p_Bd, mixed_moran_abs_time(G, r=r, p_Bd=p_Bd))
    for p_Bd in np.linspace(0, 1, 10)
    for r in (1., 1.1, 1.5, 10)
  ]
  df = pd.DataFrame(data, columns=["name", "N", "r", "p_Bd", "steps"])




