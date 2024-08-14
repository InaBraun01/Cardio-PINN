#!/bin/bash

# Define the parameter name
factor_aiso="factor_aiso"
factor_af="factor_af"
factor_as="factor_as"
factor_afs="factor_afs"
factor_biso="factor_biso"
factor_bf="factor_bf"
factor_bs="factor_bs"
factor_bfs="factor_bfs"

# Define the range for the random float
min=0.0
max=0.4

# Number of times to run the script
num_runs=30

# Loop for the specified number of runs
for ((i=1; i<=$num_runs; i++))
do
    # Generate a random float between min and max using Python
    random_float_1a=$(python -c 'import random; print(random.uniform('"$min"', '"$max"'))')
    random_float_2a=$(python -c 'import random; print(random.uniform('"$min"', '"$max"'))')
    random_float_3a=$(python -c 'import random; print(random.uniform('"$min"', '"$max"'))')
    random_float_4a=$(python -c 'import random; print(random.uniform('"$min"', '"$max"'))')
    random_float_1b=$(python -c 'import random; print(random.uniform('"$min"', '"$max"'))')
    random_float_2b=$(python -c 'import random; print(random.uniform('"$min"', '"$max"'))')
    random_float_3b=$(python -c 'import random; print(random.uniform('"$min"', '"$max"'))')
    random_float_4b=$(python -c 'import random; print(random.uniform('"$min"', '"$max"'))')
    
    echo "Run $i: Running Python script with follwing factors a_iso= $random_float_1a, a_f= $random_float_2a, a_s= $random_float_3a, a_fs= $random_float_4a, b_iso= $random_float_1b, b_f= $random_float_2b, b_s= $random_float_3b, b_fs= $random_float_4b"
    python CardioPINN_diastole.py --$factor_aiso $random_float_1a --$factor_af $random_float_2a --$factor_as $random_float_3a --$factor_afs $random_float_4a --$factor_biso $random_float_1b --$factor_bf $random_float_2b --$factor_bs $random_float_3b --$factor_bfs $random_float_4b
    echo "----------------------------------------"
done
