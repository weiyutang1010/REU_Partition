import sys
# "/nfs/guille/huang/users/tangwe/Qx/n15_y.txt"
with open("/nfs/guille/huang/users/tangwe/Qx/n16_y_emp.txt", 'r') as f:
    print("(((((.....)))))")

    lines = f.read().split('\n')
    for line in lines:
        print(line.split('\t')[-1], line.split(' ')[0])