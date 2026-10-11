"""TinyTensor 在 MNIST 上训练 CNN。用法: python mnist_cnn.py [--epochs 10] [--no-graph] [--data MNIST/raw 目录]"""
import _paths  # noqa: F401
from mnist_common import parse_args, run
from tiny_trainer import TinyTrainer

if __name__ == "__main__":
    args = parse_args("cnn", default_optimizer="sgd", default_lr=0.1)
    args.graph = True if args.graph is None else args.graph
    trainer = TinyTrainer("cnn", args.seed, export_init=True, use_graph=args.graph, chunk=args.chunk)
    run("cnn", trainer, args, framework="TinyTensor")
