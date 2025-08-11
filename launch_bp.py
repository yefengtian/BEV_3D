{
    "version": "0.2.0",
    "configurations": [
      {
        "name": "① 单卡调试（不分布式）",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/train_bev3d.py",
        "console": "integratedTerminal",
        "justMyCode": false,
        "args": [
          "${workspaceFolder}/your_config.py",
          "--launcher", "none",
          "--work-dir", "${workspaceFolder}/work_dirs/debug_single",
          "--gpus", "1"
        ],
        "env": {
          "CUDA_VISIBLE_DEVICES": "0",
          "OMP_NUM_THREADS": "1",
          "MKL_NUM_THREADS": "1"
        }
      },
      {
        "name": "② 多卡训练（torchrun，单机N卡）",
        "type": "python",
        "request": "launch",
        "module": "torch.distributed.run",
        "console": "integratedTerminal",
        "justMyCode": false,
        "args": [
          "--nproc_per_node=4",
          "--master_port=29501",
          "${workspaceFolder}/train_bev3d.py",
          "${workspaceFolder}/your_config.py",
          "--launcher", "pytorch",
          "--work-dir", "${workspaceFolder}/work_dirs/debug_ddp"
        ],
        "env": {
          "CUDA_VISIBLE_DEVICES": "0,1,2,3",
          "CUDA_DEVICE_MAX_CONNECTIONS": "1",
          "NCCL_ASYNC_ERROR_HANDLING": "1",
          "NCCL_P2P_DISABLE": "0",
          "NCCL_IB_DISABLE": "1",
          "OMP_NUM_THREADS": "1",
          "MKL_NUM_THREADS": "1"
        }
      }
    ]
  }
  