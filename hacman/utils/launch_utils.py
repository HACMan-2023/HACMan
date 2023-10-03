import os
import numpy as np
import logging
import torch

def use_freer_gpu():
    """
    Set the program to use the freer GPU. Run this at the beginning of the program.
    """
    try:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        if len(memory_available) == 1:
            return os.environ['CUDA_VISIBLE_DEVICES']

        gpu_idx = np.argmax(memory_available).item()
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_idx}'
        # torch.cuda.set_device(gpu_idx)
        print(f"Free GPU: {gpu_idx}")
        
    except:
        gpu_idx = 0
    
    return gpu_idx