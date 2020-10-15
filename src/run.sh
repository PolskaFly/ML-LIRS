traces=(2_pools cpp cs gli multi1 multi2 multi3 ps sprite zigzag)
# traces=(gli)
for i in "${traces[@]}"
do
        # python3 lirs.py $i
        # python3 ml_lirs.py $i
        python3 in_stack_miss_graph.py $i
        python3 miss_ratio_graph.py $i
done