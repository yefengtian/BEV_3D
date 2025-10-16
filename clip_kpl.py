import pickle, os
src = "data/carla_bev/0801_all_51725.pkl"   # 改成你的 val/test pkl
dst = "data/carla_bev/0801_5_samples.pkl"
N = 5
with open(src, "rb") as f:
   d = pickle.load(f)
# 兼容两种结构：list 或 dict(data_list=...)
if isinstance(d, dict) and "data_list" in d:
   d["data_list"] = d["data_list"][:N]
elif isinstance(d, list):
   d = d[:N]
else:
   raise TypeError(f"Unknown ann format: {type(d)} with keys={getattr(d,'keys',lambda:[])()}")
with open(dst, "wb") as f:
   pickle.dump(d, f)
print("Wrote:", dst, "size:", N)