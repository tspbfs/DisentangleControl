

pairs=(
  "0.25  0.01"
  "0.75  0.04"
  "1  0.05"
  "4  0.05"
)

for i in 1 2 3 4 5
do
    for pair in "${pairs[@]}"; 
    do 
        set -- $pair
        sc=$1
        ba=$2
        echo "$i $sc $ba"
        CUDA_VISIBLE_DEVICES=1 python tracking_exp_code.py --prefix $i --scale $sc --beta $ba --gradstep 5 --T 240
    done
done

