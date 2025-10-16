# model_interface/mmdet3d/hooks/val_loss_hook.py
import math
import os.path as osp
import torch
from mmcv.runner import HOOKS, Hook
from mmcv.parallel import is_module_wrapper


def parse_losses(losses):
    """Parse losses dict and return total loss and log_vars.
    
    Args:
        losses (dict): Dictionary of losses.
        
    Returns:
        tuple: (total_loss, log_vars)
    """
    log_vars = dict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items()
               if 'loss' in _key)
    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            loss_value = loss_value.data.clone()
            torch.distributed.all_reduce(loss_value.div_(torch.distributed.get_world_size()))
        log_vars[loss_name] = loss_value.item()
    return loss, log_vars


@HOOKS.register_module()
class ValLossHook(Hook):
    """在验证集上计算平均 loss，并按最小 val_loss 保存 best checkpoint。"""

    def __init__(self,
                 dataset_cfg: dict,
                 dataloader_cfg: dict,
                 interval: int = 1,
                 rule: str = 'less',
                 filename_tmpl: str = 'best_val_loss_epoch_{:03d}.pth'):
        """
        Args:
            dataset_cfg: 用于构建 val 数据集的 cfg（必须是训练式 pipeline + test_mode=False）
            dataloader_cfg: 用于构建 DataLoader 的参数（samples_per_gpu、workers_per_gpu…）
            interval: 每多少个 epoch 评估一次
            rule: 'less' 表示越小越好
            filename_tmpl: 最优 ckpt 的文件名模板
        """
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.interval = interval
        assert rule in ('less', 'greater')
        self.rule = rule
        self.filename_tmpl = filename_tmpl

        self._dataloader = None
        self.best = math.inf if rule == 'less' else -math.inf

    def before_run(self, runner):
        # 延迟构建，避免 import 顺序问题
        from mmdet.datasets import build_dataloader
        from model_interface.mmdet3d.datasets import build_dataset

        dataset = build_dataset(self.dataset_cfg)
        self._dataloader = build_dataloader(dataset=dataset, **self.dataloader_cfg)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        assert self._dataloader is not None

        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.eval()

        total_loss, total_samples = 0.0, 0
        with torch.no_grad():
            for data in self._dataloader:
                # 与训练一致：return_loss=True，得到各项 loss，再 parse 汇总
                losses = model(return_loss=True, **data)
                loss, log_vars = parse_losses(losses)

                # 批大小：img_metas 是 list[dict] 的 DataContainer
                bs = 1
                if 'img_metas' in data and hasattr(data['img_metas'], 'data'):
                    bs = len(data['img_metas'].data[0])
                total_loss += float(loss.item()) * bs
                total_samples += bs

        val_loss = total_loss / max(1, total_samples)
        # 打到日志里
        runner.log_buffer.output['val_loss'] = val_loss
        runner.log_buffer.ready = True

        # 是否更优
        improved = (val_loss < self.best) if self.rule == 'less' else (val_loss > self.best)
        if improved:
            self.best = val_loss
            filename = self.filename_tmpl.format(runner.epoch + 1)
            runner.save_checkpoint(
                runner.work_dir,
                filename_tmpl=filename,
                create_symlink=False)

            # 记录到 meta（兼容一些上游工具读取 best）
            if runner.meta is None:
                runner.meta = {}
            runner.meta.setdefault('hook_msgs', {})
            runner.meta['hook_msgs']['best_score'] = val_loss
            runner.meta['hook_msgs']['best_ckpt'] = osp.join(runner.work_dir, filename)
