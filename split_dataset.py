import mmcv, numpy as np

src = 'data/carla_bev/0801_all_51725.pkl'
dst_train = 'data/carla_bev/0801_all_51725_train_v1.pkl'
dst_val   = 'data/carla_bev/0801_all_51725_val_v1.pkl'
dst_test  = 'data/carla_bev/0801_all_51725_test_v1.pkl'  # 可选

data = mmcv.load(src, file_format='pkl')
idx = np.arange(len(data))
# np.random.seed(42)
# np.random.shuffle(idx)

n = len(idx)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
train_idx = idx[:n_train]
val_idx   = idx[n_train:n_train+n_val]
test_idx  = idx[n_train+n_val:]

mmcv.dump([data[i] for i in train_idx], dst_train)
mmcv.dump([data[i] for i in val_idx],   dst_val)
mmcv.dump([data[i] for i in test_idx],  dst_test)
print(len(train_idx), len(val_idx), len(test_idx))
