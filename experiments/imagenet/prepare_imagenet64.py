"""
把 Hugging Face 上的 ImageNet-1k 64x64（benjamin-paine/imagenet-1k-64x64：中心裁成正方形后 Lanczos 缩到 64x64，
parquet 里每张图是一段编码后的图片字节，label 0~999）转成和 prepare_tiny_imagenet.py 相同格式的 .npy：
    train_x.npy [1281167, 64, 64, 3] uint8, train_y.npy [1281167] int32
    val_x.npy   [50000, 64, 64, 3]   uint8, val_y.npy   [50000]   int32
训练集约 15.7GB，先用 open_memmap 写盘，不需要一次放进内存。

用法：python prepare_imagenet64.py --parquet <下载的 data/ 目录> --out <输出目录>
"""
import argparse
import glob
import io
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def decode(blobs):
    out = np.empty((len(blobs), 64, 64, 3), np.uint8)
    for i, b in enumerate(blobs):
        im = Image.open(io.BytesIO(b)).convert("RGB")
        if im.size != (64, 64):
            im = im.resize((64, 64), Image.LANCZOS)
        out[i] = np.asarray(im)
    return out


def convert(files, out_dir, name, workers):
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    x = np.lib.format.open_memmap(os.path.join(out_dir, name + "_x.npy"), mode="w+", dtype=np.uint8, shape=(n, 64, 64, 3))
    y = np.empty(n, np.int32)
    pos = 0
    with Pool(workers) as pool:
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=8192, columns=["image", "label"]):
                imgs = batch.column("image").to_pylist()
                labels = np.asarray(batch.column("label").to_pylist(), np.int32)
                blobs = [im["bytes"] for im in imgs]
                chunks = [blobs[i:i + 512] for i in range(0, len(blobs), 512)]
                for arr in pool.map(decode, chunks):
                    x[pos:pos + len(arr)] = arr
                    pos += len(arr)
                y[pos - len(labels):pos] = labels
            print(name, f, pos, "/", n, flush=True)
    assert pos == n
    x.flush()
    np.save(os.path.join(out_dir, name + "_y.npy"), y)
    print(name, x.shape, "labels", y.min(), y.max(), "classes", len(np.unique(y)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    convert(sorted(glob.glob(os.path.join(args.parquet, "validation-*.parquet"))), args.out, "val", args.workers)
    convert(sorted(glob.glob(os.path.join(args.parquet, "train-*.parquet"))), args.out, "train", args.workers)


if __name__ == "__main__":
    main()
