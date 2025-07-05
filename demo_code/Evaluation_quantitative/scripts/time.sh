CUDA_VISIBLE_DEVICES=1 nohup python -u eval_nr_org.py >> logs/time/nr_20240616.log 2>&1 & # PC  

CUDA_VISIBLE_DEVICES=1 nohup python -u eval_fr.py >> logs/time/fr_20240616.log 2>&1 & # PC  


#
CUDA_VISIBLE_DEVICES=1 python -u eval_nr_org.py

CUDA_VISIBLE_DEVICES=1 python -u eval_nr_org.py