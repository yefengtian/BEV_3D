# projects/freespace/hooks/throughput_hook.py
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook
import time

@HOOKS.register_module()
class ThroughputHook(Hook):
   """在 log 里增加 samples_per_sec，用于训练吞吐 KPI."""
   def __init__(self,dataloader_cfg):
       # 可以从 cfg.data.samples_per_gpu 自动读，也可以手动传
       self.samples_per_gpu = dataloader_cfg.samples_per_gpu
       self.world_size = 1
   def before_run(self, runner):
       if self.samples_per_gpu is None:
           cfg = runner.cfg
           # 你这套 BEVDepth/BEV_3D 基本是 mmcv1.x 写法
           self.samples_per_gpu = cfg.samples_per_gpu
       # 获取实际 GPU 数量
       if dist.is_available() and dist.is_initialized():
           self.world_size = dist.get_world_size()
       else:
           self.world_size = 1

   def before_train_epoch(self,runner):
       self._prev_time = time.time()

   def after_train_iter(self, runner):
       log_buffer = runner.log_buffer
       # mmdet 的 loggerHook 已经把 'time' 算好放到 output 里了
       now = time.time()
       if self._prev_time is None:
           self._prev_time = now
           return
       
       iter_time = now -self._prev_time
       self._prev_time = now
    
       if iter_time <= 0:
           return
       samples = self.samples_per_gpu * self.world_size
       ips = samples / iter_time  # images/s 或 samples/s
       # 写回 log_buffer，让 TextLoggerHook 一起打印
       log_buffer.output['samples_per_sec'] = ips
       log_buffer.update(dict(samples_per_sec = ips),count=1)