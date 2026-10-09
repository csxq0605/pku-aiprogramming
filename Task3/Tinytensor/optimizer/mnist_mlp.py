from mnist_common import parse_args, run
from tiny_nn import MLP

if __name__ == "__main__":
    args = parse_args("mlp", default_optimizer="sgd", default_lr=0.1)
    args.graph = True if args.graph is None else args.graph
    run("mlp", MLP(args.seed, export_init=True, use_graph=args.graph, chunk=args.chunk), args, framework="TinyTensor")
