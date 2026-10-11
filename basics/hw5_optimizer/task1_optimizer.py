"""
本文件我们尝试实现一个Optimizer类，用于优化一个简单的双层Linear Network
本次作业主要的内容将会在opti_epoch内对于一个epoch的参数进行优化
分为SGD_epoch和Adam_epoch两个函数，分别对应SGD和Adam两种优化器
其余函数为辅助函数，也请一并填写
和大作业的要求一致，我们不对数据处理和读取做任何要求
因此你可以引入任何的库来帮你进行数据处理和读取
理论上我们也不需要依赖lab5的内容，如果你需要的话，你可以将lab5对应代码copy到对应位置
"""
# from task0_autodiff import *
# from task0_operators import *
import sys

import numpy as np
import torch
from torchvision import datasets, transforms
#from torch.utils.data import DataLoader, Dataset

# class MNISTDataset(Dataset):
#     def __init__(self, data1, data2, data3, data4, transform=None):
#         self.data1 = data1
#         self.data2 = data2
#         self.data3 = data3
#         self.data4 = data4
#         self.transform = transform

#     def __len__(self):
#         return len(self.data1)

#     def __getitem__(self, idx):
#         sample1 = self.data1[idx]
#         sample2 = self.data2[idx]
#         sample3 = self.data3[idx % 10000]
#         sample4 = self.data4[idx % 10000]
#         return sample1, sample2, sample3, sample4

def parse_mnist():
    """
    读取MNIST数据集，并进行简单的处理，如归一化
    你可以可以引入任何的库来帮你进行数据处理和读取
    所以不会规定你的输入的格式
    但需要使得输出包括X_tr, y_tr和X_te, y_te
    """
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    root = sys.argv[1] if len(sys.argv) > 1 else './data'

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_labels = next(iter(train_loader))
    train_data = train_data.numpy().reshape(train_data.shape[0], -1)
    train_labels = train_labels.numpy()

    test_data, test_labels = next(iter(test_loader))
    test_data = test_data.numpy().reshape(test_data.shape[0], -1)
    test_labels = test_labels.numpy()

    return (train_data, train_labels), (test_data, test_labels)

def set_structure(n, hidden_dim, k):
    """
    定义你的网络结构，并进行简单的初始化
    一个简单的网络结构为两个Linear层，中间加上ReLU
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix.
    Example:
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return list(W1, W2)
    """

    w1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    w2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return [w1, w2]

def forward(X, weights):
    """
    使用你的网络结构，来计算给定输入X的输出
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    Example:
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2
    """
    w1 = weights[0]
    w2 = weights[1]
    h = np.maximum(X @ w1, 0)
    z = h @ w2
    return z, h

def softmax(Z):
    """按行做 softmax（先减去每行最大值，避免 exp 溢出）"""
    z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return z_exp / np.sum(z_exp, axis=1, keepdims=True)

def softmax_loss(Z, y):
    """ 
    一个写了很多遍的Softmax loss...

    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    batch_size = Z.shape[0]
    logits = Z - np.max(Z, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits), axis=1))
    return float(np.mean(log_sum_exp - logits[np.arange(batch_size), y]))

def opti_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999, using_adam=False, state=None):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    state: Adam 的时间步与一、二阶矩，跨 epoch 保留
    """
    if using_adam:
        Adam_epoch(X, y, weights, lr = lr, batch=batch, beta1=beta1, beta2=beta2, state=state)
    else:
        SGD_epoch(X, y, weights, lr = lr, batch=batch)

def compute_grad(Z, H, X_batch, y_batch, w2, w1, batch):
    """
    计算梯度（Z 为 logits）
    """
    batch = Z.shape[0]
    P = softmax(Z)
    y_onehot = np.zeros_like(P)
    y_onehot[np.arange(batch), y_batch] = 1
    dZ = (P - y_onehot) / batch
    grad_w2 = H.T @ dZ
    dH = dZ @ w2.T
    dH[H <= 0] = 0
    grad_w1 = X_batch.T @ dH
    return grad_w1, grad_w2

def SGD_epoch(X, y, weights, lr = 0.1, batch=100):
    """ 
    SGD优化一个List of Weights
    本函数应该inplace地修改Weights矩阵来进行优化
    用学习率简单更新Weights

    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    num_examples = X.shape[0]
    # indices = np.random.permutation(num_examples)
    # X_indices, y_indices = X[indices], y[indices]
    for i in range(0, num_examples, batch):
        X_batch, y_batch = X[i:i+batch], y[i:i+batch]
        Z, H = forward(X_batch, weights)
        w1, w2 = weights[0], weights[1]
        #print(w2)
        grad_w1, grad_w2 = compute_grad(Z, H, X_batch, y_batch, w2, w1, batch) 
        w1 -= lr * grad_w1
        w2 -= lr * grad_w2
        #print(w2)

def Adam_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999, state=None):
    r"""
    ADAM优化一个
    本函数应该inplace地修改Weights矩阵来进行优化
    使用Adaptive Moment Estimation来进行更新Weights
    具体步骤可以是：
    1. 增加时间步 $t$。
    2. 计算当前梯度 $g$。
    3. 更新一阶矩向量：$m = \beta_1 \cdot m + (1 - \beta_1) \cdot g$。
    4. 更新二阶矩向量：$v = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$。
    5. 计算偏差校正后的一阶和二阶矩估计：$\hat{m} = m / (1 - \beta_1^t)$ 和 $\hat{v} = v / (1 - \beta_2^t)$。
    6. 更新参数：$\theta = \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$。
    其中$\eta$表示学习率，$\beta_1$和$\beta_2$是平滑参数，
    $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数，如1e-8。
    
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum
        state (dict): 时间步 t 与一、二阶矩；传入同一个 dict 时这些量跨 epoch 保留，
            不传则每个 epoch 从 0 开始（偏差校正和矩估计都会被重置）

    Returns:
        None
    """
    if state is None:
        state = {}
    if not state:
        state.update(t=0, m=[np.zeros_like(w) for w in weights], v=[np.zeros_like(w) for w in weights])
    t = state["t"]
    (m_w1, m_w2), (v_w1, v_w2) = state["m"], state["v"]
    num_examples = X.shape[0]
    for i in range(0, num_examples, batch):
        t += 1
        X_batch, y_batch = X[i:i+batch], y[i:i+batch]
        Z, H = forward(X_batch, weights)
        w1, w2 = weights[0], weights[1]
        grad_w1, grad_w2 = compute_grad(Z, H, X_batch, y_batch, w2, w1, batch)
        m_w1 = beta1 * m_w1 + (1 - beta1) * grad_w1
        m_w2 = beta1 * m_w2 + (1 - beta1) * grad_w2
        v_w1 = beta2 * v_w1 + (1 - beta2) * (grad_w1 ** 2)
        v_w2 = beta2 * v_w2 + (1 - beta2) * (grad_w2 ** 2)
        m_w1_hat = m_w1 / (1 - beta1 ** t)
        m_w2_hat = m_w2 / (1 - beta1 ** t)
        v_w1_hat = v_w1 / (1 - beta2 ** t)
        v_w2_hat = v_w2 / (1 - beta2 ** t)
        w1 -= lr * m_w1_hat / (np.sqrt(v_w1_hat) + 1e-8)
        w2 -= lr * m_w2_hat / (np.sqrt(v_w2_hat) + 1e-8)
    state.update(t=t, m=[m_w1, m_w2], v=[v_w1, v_w2])


def loss_err(h,y):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """ 
    训练过程
    """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    weights = set_structure(n, hidden_dim, k)
    state = {}
    # dataset = MNISTDataset(X_tr, y_tr, X_te, y_te)
    # dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        # for x_tr_batch, y_tr_batch, x_te_batch, y_te_batch in dataloader:
        #     x_tr_batch, y_tr_batch, x_te_batch, y_te_batch = x_tr_batch.numpy(), y_tr_batch.numpy(), x_te_batch.numpy(), y_te_batch.numpy()
        #     opti_epoch(x_tr_batch, y_tr_batch, weights, lr=lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam)
        # tr_indices, te_indices = np.random.permutation(X_tr.shape[0]), np.random.permutation(X_te.shape[0])
        # x_tr_batch, y_tr_batch, x_te_batch, y_te_batch = X_tr[tr_indices[:batch]], y_tr[tr_indices[:batch]], X_te[te_indices[:batch]], y_te[te_indices[:batch]]
        # train_loss, train_err = loss_err(forward(x_tr_batch, weights)[0], y_tr_batch)
        # test_loss, test_err = loss_err(forward(x_te_batch, weights)[0], y_te_batch)
        opti_epoch(X_tr, y_tr, weights, lr=lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam, state=state)
        train_loss, train_err = loss_err(forward(X_tr, weights)[0], y_tr)
        test_loss, test_err = loss_err(forward(X_te, weights)[0], y_te)
        #print(weights[1])
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
            .format(epoch + 1, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    (X_tr, y_tr), (X_te, y_te) = parse_mnist()
    weights = set_structure(X_tr.shape[1], 100, y_tr.max() + 1)
    ## using SGD optimizer 
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 0.2, batch=100, beta1=0.9, beta2=0.999, using_adam=False)
    ## using Adam optimizer
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 0.001, batch=100, beta1=0.9, beta2=0.999, using_adam=True)
    