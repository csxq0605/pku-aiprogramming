"""
把 Tiny ImageNet（tiny-imagenet-200.zip 解压后的目录）转成 uint8 NHWC 的 .npy，训练时整个数据集常驻 GPU：
    train_x.npy [100000, 64, 64, 3] uint8, train_y.npy [100000] int32
    val_x.npy   [10000, 64, 64, 3]  uint8, val_y.npy   [10000]  int32
官方测试集没有标签，按惯例用 val 作为测试集。
用法: python prepare_tiny_imagenet.py ~/data/tiny-imagenet-200
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image


def load(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def load_all(paths):
    with ThreadPoolExecutor(16) as pool:
        return np.stack(list(pool.map(load, paths)))


def main(root):
    with open(os.path.join(root, "wnids.txt")) as f:
        wnids = [line.strip() for line in f if line.strip()]
    label_of = {w: i for i, w in enumerate(wnids)}

    paths, labels = [], []
    for w in wnids:
        folder = os.path.join(root, "train", w, "images")
        for name in sorted(os.listdir(folder)):
            paths.append(os.path.join(folder, name))
            labels.append(label_of[w])
    train_x, train_y = load_all(paths), np.array(labels, dtype=np.int32)

    paths, labels = [], []
    with open(os.path.join(root, "val", "val_annotations.txt")) as f:
        for line in f:
            name, w = line.split("\t")[:2]
            paths.append(os.path.join(root, "val", "images", name))
            labels.append(label_of[w])
    val_x, val_y = load_all(paths), np.array(labels, dtype=np.int32)

    for name, arr in (("train_x", train_x), ("train_y", train_y), ("val_x", val_x), ("val_y", val_y)):
        np.save(os.path.join(root, name + ".npy"), arr)
        print(name, arr.shape, arr.dtype)
    # 分块统计，避免把 1.2GB 的 uint8 一次性转成 float64
    s, s2, n = np.zeros(3), np.zeros(3), 0
    for i in range(0, len(train_x), 5000):
        chunk = train_x[i:i + 5000].reshape(-1, 3).astype(np.float64) / 255
        s, s2, n = s + chunk.sum(0), s2 + (chunk ** 2).sum(0), n + len(chunk)
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    print("mean", np.round(mean, 4), "std", np.round(std, 4))


if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/data/tiny-imagenet-200"))
