"""把仓库根目录（import tinytensor）与各实验目录（脚本之间用平铺的模块名互相 import）加入 sys.path"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(ROOT) == "experiments":
    ROOT = os.path.dirname(ROOT)
for _d in ("", "experiments/mnist", "experiments/imagenet"):
    _d = os.path.join(ROOT, *_d.split("/")) if _d else ROOT
    if _d not in sys.path:
        sys.path.insert(0, _d)
