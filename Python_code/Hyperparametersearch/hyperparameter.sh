#!/bin/bash

# Define the parameter name
max_act="max_act"
Windkessel_R="Windkessel_R"
Windkessel_C="Windkessel_C"

# Define the range for the random float
min_Ta=0.85e5
max_Ta=2.5e5

min_R=5.0
max_R=50.0

min_C=5.0e-7
max_C=5.0e-6
# Define output file
output_file="Hyperparmeters_act_max_C_R.txt"

# Clear the output file if it exists
> "$output_file"

# Number of times to run the script
num_runs=1

# Loop for the specified number of runs
for ((i=1; i<=$num_runs; i++))
do
    # Generate a random float between min and max using Python
    random_float_1=$(python -c "import random; print(random.uniform($min_Ta, $max_Ta))")
    random_float_2=$(python -c "import random; print(random.uniform($min_C, $max_C))")
    random_float_3=$(python -c "import random; print(random.uniform($min_R, $max_R))")
    
    echo "Run $i: Running Python script with maximum active stress = $random_float_1, C = $random_float_2, R= $random_float_3" | tee -a "$output_file"
    python Hyperparametersearch/CardioPINN_bash_Ta.py --$max_act $random_float_1 #--$Windkessel_C $random_float_2 --$Windkessel_R $random_float_3
    echo "----------------------------------------"
done

echo "Output has been saved to $output_file"
