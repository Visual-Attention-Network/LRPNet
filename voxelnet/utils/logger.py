from .general import build_file, current_time
from .registry import HOOKS,build_from_cfg 
import time 
import os
from tensorboardX import SummaryWriter
from .config import get_cfg


@HOOKS.register_module()
class RunLogger:
    def __init__(self,work_dir,rank=0):
        self.rank = rank
        if self.rank>0:
            return 
        self.writer = SummaryWriter(os.path.join(work_dir,"tensorboard"),flush_secs=10)
        save_file = build_file(work_dir,prefix="textlog/log_"+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+".txt")
        self.log_file = open(save_file,"a")

    def log(self,data,global_step,**kwargs):
        if self.rank>0:
            return 
        data.update(kwargs)
        data = {k:d.item() if hasattr(d,"item") else d for k,d in data.items()}
        msg = f"iter: {global_step}"
        for k,d in data.items():
            msg += f", {k}: {d:4f}" if isinstance(d,float) else f", {k}: {d}"
            if isinstance(d,str):
                continue
            self.writer.add_scalar(k,d,global_step=global_step)
        self.print(msg)
    
    def print(self,msg):
        if self.rank>0:
            return 
        now_time = current_time()
        msg = now_time+", "+msg
        self.log_file.write(msg+"\n")
        self.log_file.flush()
        print(msg)