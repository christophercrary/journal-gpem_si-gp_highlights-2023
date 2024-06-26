from itertools import product
import locale
import os
import pickle
import re
import subprocess

import numpy as np

# Set locale, as described in the following link:
# https://tinyurl.com/w29vbafe
locale.setlocale(locale.LC_ALL, '')

# Useful path directory.
# root_dir = (f'{os.getcwd()}/../../../../results/energies/operon')
root_dir = (f'{os.getcwd()}/../../../../results/energies/operon')

# Primitive set names.
primitive_set_names = ('A', 'B', 'C')
# primitive_set_names = ('C',)

# Number of fitness cases.
n_fitness_cases = (10, 100, 1000, 10000, 100000)
# n_fitness_cases = (100000,)

# Number of energy measurements.
n = 11
# n = 3

# Dictionary to keep track of energy/runtime results for each 
# combination of primitive set and number of fitness cases.
results = {
    f'nicolau_{p.lower()}' : {
        f : {} for f in n_fitness_cases}
    for p in primitive_set_names}

# Compute results.
for (p, f) in product(primitive_set_names, n_fitness_cases):
    # Temporary lists of energy/runtime results
    # for the current combination.
    energy = []
    runtime = []
    # Perform `n` measurements.
    for _ in range(n):
        text = subprocess.check_output(
            f'sudo perf stat -e power/energy-pkg/ ./operon-test --tc="{p}{f}"', 
            shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        # Extract energy of the individual run.
        m = re.search('( +)(.+?) Joules', text)
        energy.append(float(locale.atof(m.group(2))))
        # Extract runtime of the individual run.
        m = re.search('( +)(.+?) seconds time elapsed', text)
        runtime.append(float(locale.atof(m.group(2))))
    # Convert to NumPy arrays.
    energy = np.array(energy)
    runtime = np.array(runtime)
    # Preserve energy results for the current combination
    # of primitive set and number of fitness cases.
    p = f'nicolau_{p.lower()}'
    results[p][f]['energy'] = energy
    results[p][f]['energy_mean'] = np.mean(energy)
    results[p][f]['energy_median'] = np.median(energy)
    results[p][f]['energy_max'] = np.max(energy)
    results[p][f]['energy_min'] = np.min(energy)
    results[p][f]['runtime'] = runtime
    results[p][f]['runtime_mean'] = np.mean(runtime)
    results[p][f]['runtime_median'] = np.median(runtime)
    results[p][f]['runtime_max'] = np.max(runtime)
    results[p][f]['runtime_min'] = np.min(runtime)

for (p, f) in product(primitive_set_names, n_fitness_cases):
    p = f'nicolau_{p.lower()}'
    print(
        f'Median energy, {p}, {f} fitness cases : '
        f'{results[p][f]["energy_median"]}')
    
with open(f'{root_dir}/results.pkl', 'wb') as f:
    pickle.dump(results, f)