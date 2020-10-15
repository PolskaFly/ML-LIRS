# traces=(2_pools cpp cs gli multi1 multi2 multi3 ps sprite w111)
traces=(gli)
for i in "${traces[@]}"
do
        cp ~/myLirs/result_set/w$i/w$i-OPT ./result_set/$i/$i-OPT
done