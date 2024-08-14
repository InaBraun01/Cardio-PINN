#!/bin/bash

# Define the parameter name
max_act="max_act"

# Define the range for the grid search
min=0.05e5
max=0.9e5

# Define the number of steps in the grid
num_steps=20

# Calculate the step size
step_size=$(python -c "print(($max - $min) / $num_steps)")

# Loop through the grid values
for i in $(seq 0 $num_steps)
do
    # Calculate the current value
    current_value=$(python -c "print($min + $i * $step_size)")
    
    echo "Run $((i+1)): Running Python script with following factors max_act= $current_value"
    python CardioPINN_systole.py --$max_act $current_value
    echo "----------------------------------------"
done