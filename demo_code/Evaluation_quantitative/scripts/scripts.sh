
##### NR 
CUDA_VISIBLE_DEVICES=1 nohup python -u eval_nr.py > logs/eval_nr_naturality.log 2>&1 & # PC 2641210




CUDA_VISIBLE_DEVICES=0 nohup python -u eval_nr.py > logs/eval_nr_aesthetics.log 2>&1 & # PC 2882325

 
# runing 
CUDA_VISIBLE_DEVICES=1 nohup python -u eval_nr.py > logs/eval_nr_quality_2.log 2>&1 & # PC 663219