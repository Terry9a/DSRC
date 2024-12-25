import os   

for index in range(19,27,2):
  cmd = f"CUDA_VISIBLE_DEVICES=2  python /home/gaojing/zjy/code/opencood/tools/inference.py --eval_epoch {index} "
  print(f"Running command: {cmd}")
  os.system(cmd)