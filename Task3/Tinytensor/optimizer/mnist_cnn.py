from mnist_common import parse_args, run
from tiny_nn import CNN

if __name__ == "__main__":
    args = parse_args("cnn", default_optimizer="sgd", default_lr=0.1)
    args.graph = True if args.graph is None else args.graph
    run("cnn", CNN(args.seed, export_init=True, use_graph=args.graph, chunk=args.chunk), args, framework="TinyTensor")
