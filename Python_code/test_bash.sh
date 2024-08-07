#!/bin/bash

# Define the parameter name
param_name="a"
param_name_1="b"

# Define the range for the random float
min=0.1
max=10.0

# Number of times to run the script
num_runs=5

# Loop for the specified number of runs
for ((i=1; i<=$num_runs; i++))
do
    # Generate a random float between min and max using Python
    random_float=$(python -c "import random; print(random.uniform($min, $max))")
    random_float_2=$(python -c "import random; print(random.uniform($min, $max))")
    
    echo "Run $i: Running Python script with $param_name = $random_float and $param_name_1 = $random_float_2"
    python test_bash.py --$param_name $random_float --$param_name_1 $random_float_2
    echo "----------------------------------------"
done