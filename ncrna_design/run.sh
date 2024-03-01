#!/bin/bash

curr_id=26
structs=("(((((......)))))" "..((((((((.....)).)))))).."  ".(((((((((((...)))))....)))))).")

for var in "${structs[@]}"; do
    echo "Current variable: $var"

    # python Qx_nussinov.py $var &

    # echo $var | python rna_design.py --step 2500 --sharpturn 3 --penalty 0 --lr 0.005 --k 1000 --id $curr_id --obj pyx_sampling_Dy
    python analysis.py --file $curr_id --mode graph > graphs/$curr_id.txt
    (( curr_id++ ))

    # echo $var | python rna_design.py --step 2500 --sharpturn 3 --penalty 0 --lr 0.01 --id $curr_id --obj pyx_jensen_Dy
    python analysis.py --file $curr_id --mode graph > graphs/$curr_id.txt
    (( curr_id++ ))
done