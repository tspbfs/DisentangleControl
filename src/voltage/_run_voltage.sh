
for i in 1 2 3 4 5
do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python voltage_exp_code.py --prefix $i --scale 1.25 --beta 0.08 --gradstep 50 --Buffer 50 --T 200
done



