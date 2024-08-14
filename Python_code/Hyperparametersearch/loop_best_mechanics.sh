

#!/bin/bash

# Skip the header row
tail -n +2 best_results_diastole.csv | while IFS=',' read -r a_iso b_iso a_f b_f a_s b_s a_fs b_fs
do
    python3 CardioPINN_file_half.py "$a_iso" "$b_iso" "$a_f" "$b_f" "$a_s" "$b_s" "$a_fs" "$b_fs"
done
