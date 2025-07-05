


CUDA_VISIBLE_DEVICES=0 nohup python -u test_512.py -T T2I > logs/T2I.log  2>&1 & #


CUDA_VISIBLE_DEVICES=2 nohup python -u test_512.py -T I2I > logs/I2I.log  2>&1 & #


CUDA_VISIBLE_DEVICES=0 nohup python -u test_512.py -T TI2I > logs/TI2I.log  2>&1 & # 

