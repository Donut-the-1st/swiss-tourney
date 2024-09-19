import random
import numpy as np
import scipy.optimize as opt
from itertools import combinations
from typing import *


def get_pairings():
    pass


def get_pairings_random(n):
    numbers = list(range(1, n + 1)) # start at 1
    random.shuffle(numbers)
    pairings = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - 1, 2)]
    if n % 2 != 0:
        pairings.append((numbers[-1],))
    return pairings


def get_pairings_chain():
    pass


def get_pairings_lp(scores: np.ndarray, win_ratios: np.ndarray):
    possible_pairs = list(combinations(range(len(scores)), 2))
    score_deltas = [lp_cost(pair, scores, win_ratios) for pair in possible_pairs]

    constraint_matrix = np.zeros((len(scores), len(possible_pairs)))
    for [option, pair] in enumerate(possible_pairs):
        constraint_matrix[pair[0], option] = 1
        constraint_matrix[pair[1], option] = 1

    constraint_bound = np.ones(len(scores))
    constraints = opt.LinearConstraint(constraint_matrix, constraint_bound, constraint_bound)
    integrality = np.ones_like(score_deltas)

    res = opt.milp(c=score_deltas, constraints=constraints, integrality=integrality)
    pairs = list()

    if not res.success:
    	print("ERROR:", res.message)
    	return None

    for i in range(len(possible_pairs)):
        if res.x[i] == 1:
            pairs.append(possible_pairs[i])

    return pairs


def lp_cost(pair, scores, games):
    delta = abs(scores[pair[0]] - scores[pair[1]])
    # If A has won against B, or B has won against A, make that match up less likely to occor
    if games[pair[0]][pair[1]] != 0 or games[pair[1]][pair[0]] != 0:
        delta += 1
    return delta

