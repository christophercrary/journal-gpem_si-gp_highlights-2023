import datetime as dt
from functools import partial
import os
import pickle
import timeit
import sys

import numpy as np
from sklearn.metrics import mean_squared_error

import deap.gp

# sys.path.insert(1, 'experiment/tools/setup/')
sys.path.insert(1, '../setup/')
from primitive_sets import nicolau_a, nicolau_b, nicolau_c

# Useful directory path.
# root_dir = f'{os.getcwd()}/experiment/results/programs'
root_dir = f'{os.getcwd()}/../../results/programs'

# Load input/target data.
with open(f'{root_dir}/../inputs.pkl', 'rb') as f:
    inputs = pickle.load(f)
with open(f'{root_dir}/../target.pkl', 'rb') as f:
    targets = pickle.load(f)

@np.errstate(all='ignore')
def evaluate_(tree, primitive_set, nfc, n_iv):

    # Extract the relevant input/target data.
    inputs_ = inputs[:nfc, :n_iv]
    targets_ = targets[:nfc]

    # Transform `PrimitiveTree` object into a callable function.
    program = deap.gp.compile(tree, primitive_set)

    # Calculate program outputs.
    y = tuple([program(*input_) for input_ in inputs_])
    
    try:
        # Calculate and return fitness.
        return np.sqrt(mean_squared_error(targets_, y))
    except ValueError:
        return float('inf')

def evaluate(trees, primitive_set, nfc, n_iv, fitness):
    """Return list of fitness scores for programs.
    
    Root-mean square error is used as the fitness function.
    """
    fn = partial(evaluate_, primitive_set=primitive_set, nfc=nfc, n_iv=n_iv)
    # Calculate fitness scores for the set of trees.
    outputs = [fn(tree) for tree in trees]
    fitness[:] = outputs[:]

if __name__ == '__main__':

    # Primitive sets.
    primitive_sets = {
        'nicolau_a': nicolau_a,
        'nicolau_b': nicolau_b,
        'nicolau_c': nicolau_c,
    }

    # Numbers of fitness cases.
    n_fitness_cases = (10, 100, 1000, 10000, 100000)
    # n_fitness_cases = (10, 100,)

    # Number of program bins.
    n_bins = 32

    # Number of programs per bin.
    n_programs = 512

    # Number of times in which experiments are run.
    n_runs = 1

    # Runtimes for programs within each bin, for each number 
    # of fitness cases, for each function set.
    runtimes = []

    # Fitness results for each program bin, for each number
    # of fitness cases, for each function set.
    fitnesses = {name : [[[] for _ in range(n_bins)] 
        for _ in range(len(n_fitness_cases))] for name in primitive_sets}

    for name, ps in primitive_sets.items():
        # Prepare for statistics relevant to the primitive set.
        runtimes.append([])

        # Read in the programs relevant to the primitive set from file.
        # This file contains `num_size_bins * n_programs` programs.
        with open(f'{root_dir}/{name}/programs.txt', 'r') as f:
            programs = f.readlines()

        # Primitive set object for DEAP tool.
        primitive_set = deap.gp.PrimitiveSet(
            "main_", len(ps.variables), prefix="v")

        # Add functions to primitive set.
        for name_, f in ps.functions.items():
            primitive_set.addPrimitive(f, ps.arity(name_))

        # For each amount of fitness cases, and for each size bin, 
        # calculate the relevant statistics.
        for i, nfc in enumerate(n_fitness_cases):
            # Prepare for statistics relevant to the 
            # numbers of fitness cases and size bins.
            runtimes[-1].append([[] for _ in range(n_bins)])

            for j in range(n_bins):
                # For each size bin...
                print(f'({dt.datetime.now().ctime()}) '
                      f'DEAP: evaluating programs '
                      f'for primitive set `{name}`, bin {j + 1}, {nfc} '
                      f'fitness cases...')

                # `PrimitiveTree` objects for size bin `j + 1`.
                trees = [deap.gp.PrimitiveTree.from_string(p, primitive_set) 
                    for p in programs[n_programs * (j) : n_programs * (j + 1)]]

                # Raw runtimes after running the `evaluate`
                # function a total of `n_runs` times.
                r = []
                for _ in range(n_runs):
                    fitnesses_ = []
                    r_ = timeit.timeit(
                        'evaluate(trees, primitive_set, nfc, '
                        '  len(ps.variables), fitnesses_)',
                        globals=globals(), number=1)
                    r.append(r_)
                runtimes[-1][-1][j] = r
                fitnesses[name][i][j][:] = fitnesses_[:]

            # Preserve fitness data.
            with open(
                f'{root_dir}/{name}/{nfc}/fitness_deap_regular.csv', 
                    'w+') as f:
                for j, fitness_bin in enumerate(fitnesses[name][i]):
                    for k, value in enumerate(fitness_bin):
                        f.write(f'{str(value)}')
                        if k < len(fitness_bin) - 1:
                            f.write(f'\n')
                    if j < len(fitnesses[name][i]) - 1:
                        f.write(f'\n')

    # Preserve results.
    with open(
        f'{root_dir}/../runtimes/deap/results_deap_regular.pkl', 'wb') as f:
        pickle.dump(runtimes, f)