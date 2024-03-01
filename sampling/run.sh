#!/bin/bash

# Define the range of integers
start=7
end=10

# Iterate through the range
for ((i=start; i<=end; i++)); do
    echo "n: = $i"
    python plot_Qx.py --n $i --mode 2 &
done
echo "n: = 12"
# python plot_Qx.py --n 12 --mode 2

structs=("(((...)))" "((((...)))).")
for var in "${structs[@]}"; do
    echo "Current variable: $var"
    python plot_Qx.py --y $var --mode 2 &
    python plot_Qx.py --y $var --targeted --mode 2 &
done

structs=("(((((......)))))" "..((((((((.....)).))))))..."  ".(((((((((((...)))))....)))))).." "((((((.((((((((....))))).)).).))))))." "..((((((((.....))))((((.....))))))))..")
for var in "${structs[@]}"; do
    echo "Current variable: $var"
    python plot_Qx.py --y $var --targeted --mode 2 &
done
# # Iterate through the range
# for ((i=start; i<=end; i++)); do
#     echo "n: = $i"
#     python Qx.py --n $i
# done