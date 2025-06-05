import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random
import seaborn as sns

from dataclasses import dataclass
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Tuple

class Occupant(Enum):
  RESIDENT = 0
  MUTANT = 1
  EMPTY = 2

class Event(Enum):
  BIRTH = 0
  DEATH = 1

class Outcome(Enum):
  FIXATION = 0
  EXTINCTION = 1

class Strength(Enum):
  WEAK = 0
  STRONG = 1

@dataclass
class Result:
  outcome: Outcome
  strength: Strength


def fitness(occupant: Occupant, r: float) -> float:
  return (
    r if occupant == Occupant.MUTANT else
    1. if occupant == Occupant.RESIDENT else
    0.
  )

def step(G: nx.Graph, r: float, occupants: Dict[Any, Occupant]) -> None:
  possibilities = list(itertools.chain(*[
    [
      ((Event.BIRTH, location), fitness(occupant, r)),
      ((Event.DEATH, location), int(occupant != Occupant.EMPTY)),
    ]
    for location, occupant in occupants.items()
  ]))

  population, weights = zip(*possibilities)
  event, location = random.choices(population, weights)[0]
  occupant = occupants[location]

  assert occupant != Occupant.EMPTY

  if event == Event.DEATH:
    occupants[location] = Occupant.EMPTY
    return

  # We know that the event is BIRTH.
  possible_birth_locations = [
    neighboring_location
    for neighboring_location in G.neighbors(location)
    if occupants[neighboring_location] == Occupant.EMPTY
  ]

  if len(possible_birth_locations) == 0: return

  birth_location = random.choice(possible_birth_locations)
  occupants[birth_location] = occupant


def simulate(G: nx.Graph, r: float) -> Result:
  occupants = {
    location: Occupant.RESIDENT
    for location in G.nodes
  }
  occupants[random.choice(list(G.nodes))] = Occupant.MUTANT

  fixation = None
  while True:
    step(G, r, occupants)
    distinct_occupants = set(occupants.values())
    if distinct_occupants == {Occupant.MUTANT}: return Result(Outcome.FIXATION, Strength.STRONG)
    if distinct_occupants == {Occupant.RESIDENT}: return Result(Outcome.EXTINCTION, Strength.STRONG)
    if distinct_occupants == {Occupant.EMPTY}: break

    if distinct_occupants == {Occupant.MUTANT, Occupant.EMPTY}:
      fixation = True
    if distinct_occupants == {Occupant.RESIDENT, Occupant.EMPTY}:
      fixation = False

  assert fixation is not None

  return Result(Outcome.FIXATION if fixation else Outcome.EXTINCTION, Strength.WEAK)
    

def collect_stats(G: nx.Graph, r: float, trials: int) -> pd.DataFrame:
  results = [simulate(G, r) for _ in range(trials)]
  df = pd.DataFrame(
    data=[
      (result.outcome.value, result.strength.value)
      for result in results
    ],
    columns=['fixated', 'strength'],
  )
  return df


if __name__ == '__main__':
  TRIALS = 1_000
  N = 10
  G = nx.complete_graph(N)
  R = 10
