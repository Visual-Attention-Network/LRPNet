import os 
import glob
import warnings 
import time
import shutil 
import torch
import torch.distributed
from multiprocessing import Pool
from tqdm import tqdm
import random
import numpy as np

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_file(file,ext=None):
    if file is None:
        return False
    if not os.path.exists(file):
        warnings.warn(f"{file} is not exists")
        return False
    if not os.path.isfile(file):
        warnings.warn(f"{file} must be a file")
        return False
    if ext:
        if not os.path.splitext(file)[1] in ext:
            # warnings.warn(f"the type of {file} must be in {ext}")
            return False
    return True

def search_ckpt(work_dir):
    files = glob.glob(os.path.join(work_dir,"checkpoints/ckpt_*.pkl"))
    if len(files)==0:
        return None
    files = sorted(files,key=lambda x:int(x.split("_")[-1].split(".pkl")[0]))
    return files[-1]  

def build_file(work_dir,prefix):
    """ build file and makedirs the file parent path """
    work_dir = os.path.abspath(work_dir)
    prefixes = prefix.split("/")
    file_name = prefixes[-1]
    prefix = "/".join(prefixes[:-1])
    if len(prefix)>0:
        work_dir = os.path.join(work_dir,prefix)
    os.makedirs(work_dir,exist_ok=True)
    file = os.path.join(work_dir,file_name)
    return file 

def clean(work_dir):
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        
def multi_process(func,files,processes=1):
    if processes <=1:
        for mesh_file in tqdm(files):
            func(mesh_file)
    else:
        with Pool(processes=processes) as pool:
            r = list(tqdm(pool.imap(func, files), total=len(files)))


def current_time():
    return time.asctime( time.localtime(time.time()))

def print_network(model):
    print('#parameters', sum([x.nelement() for x in model.parameters()]))