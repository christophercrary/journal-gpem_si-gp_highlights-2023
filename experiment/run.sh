# Navigate to the `tools` directory.
cd tools

# Profile DEAP.
cd deap
python profile.py

# Profile TensorGP.
cd ../tensorgp
python profile.py

# Profile Operon.
cd ../operon/operon/build
./operon-test --tc="Node Evaluations Batch"

# After fully executing this script, some relevant statistics 
# can be generated by way of the `stats.ipynb` Jupyter notebook, 
# located within the `tools` directory.