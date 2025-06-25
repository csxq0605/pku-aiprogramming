import numpy as np
import torch
from torchvision import datasets, transforms
from operators import *
from myTensor import Tensor_float as tf
from myTensor import Tensor_int as ti

def parse_mnist():
    """
    读取MNIST数据集,并进行简单的处理,如归一化
    输出包括X_tr, y_tr和X_te, y_te
    """
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_labels = next(iter(train_loader))
    train_data = train_data.numpy().reshape(-1, 28*28)
    train_labels = train_labels.numpy()

    test_data, test_labels = next(iter(test_loader))
    test_data = test_data.numpy().reshape(-1, 28*28)
    test_labels = test_labels.numpy()

    return (train_data, train_labels), (test_data, test_labels)

class CNN:
    def __init__(self):
        w1 = Tensor(tf([28*28, 100], "gpu").random(0,1).mults(1 / np.sqrt(100)))
        b1 = Tensor(tf([100],"gpu").random(0,1).mults(1 / np.sqrt(100)))
        w2 = Tensor(tf([100, 10], "gpu").random(0,1).mults(1 / np.sqrt(10)))
        b2 = Tensor(tf([10],"gpu").random(0,1).mults(1 / np.sqrt(10)))
        self.weights = [w1, b1, w2, b2]

    def forward(self, X, y):
        X = relu(fc(X, self.weights[0], self.weights[1]))
        X = fc(X, self.weights[2], self.weights[3])
        X = softmaxcrossentropyloss(X, y)
        return X

    def opti_epoch(self, X, y, lr = 0.1, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
        """
        优化一个epoch
        具体请参考SGD_epoch 和 Adam_epoch的代码
        """
        if using_adam:
            self.Adam_epoch(X, y, lr = lr, batch=batch, beta1=beta1, beta2=beta2)
        else:
            self.SGD_epoch(X, y, lr = lr, batch=batch)

    def SGD_epoch(self, X, y, lr = 0.1, batch=100):
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
        for i in range(0, num_examples, batch * 10):
            X_batch, y_batch = Tensor(X[i:i+batch]), Tensor(y[i:i+batch])
            tr_output = self.forward(X_batch, y_batch)
            tr_output.backward()
            for i in range(len(self.weights)):
                grad = self.weights[i].grad.realize_cached_data()
                self.weights[i] -= Tensor(lr * grad)

    def Adam_epoch(self, X, y, lr = 0.1, batch=100, beta1=0.9, beta2=0.999):
        """ 
        ADAM优化一个List of Weights
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
        $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数,如1e-8。
        
        Args:
            X : 2D input array of size (num_examples, input_dim).
            y : 1D class label array of size (num_examples,)
            weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
            lr (float): step size (learning rate) for SGD
            batch (int): size of SGD minibatch
            beta1 (float): smoothing parameter for first order momentum
            beta2 (float): smoothing parameter for second order momentum

        Returns:
            None
        """
        t = 0
        m_w = [np.zeros_like(self.weights[i].realize_cached_data()) for i in range(len(self.weights))]
        v_w = [np.zeros_like(self.weights[i].realize_cached_data()) for i in range(len(self.weights))]
        num_examples = X.shape[0]
        for i in range(0, num_examples, batch * 10):
            t += 1
            X_batch, y_batch = Tensor(X[i:i+batch]), Tensor(y[i:i+batch])
            tr_output = self.forward(X_batch, y_batch)
            tr_output.backward()
            for i in range(len(self.weights)):
                grad = self.weights[i].grad.realize_cached_data()
                m_w[i] = beta1 * m_w[i] + (1 - beta1) * grad
                v_w[i] = beta2 * v_w[i] + (1 - beta2) * (grad ** 2)
                m_w_hat = m_w[i] / (1 - beta1 ** t)
                v_w_hat = v_w[i] / (1 - beta2 ** t)
                self.weights[i] -= Tensor(lr * m_w_hat / (np.sqrt(v_w_hat) + 1e-8))

    def train_nn(self, X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
        """ 
        训练过程
        """
        print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
        for epoch in range(epochs):
            self.opti_epoch(X_tr, y_tr, lr=lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam)
            train_loss, train_err = self.forward(Tensor(X_tr), Tensor(y_tr)).realize_cached_data()
            test_loss, test_err = self.forward(Tensor(X_te), Tensor(y_te)).realize_cached_data()
            print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
                .format(epoch + 1, train_loss, train_err, test_loss, test_err))
            
    def test_nn(self, X_te, y_te):
        """
        测试过程
        """
        test_loss, test_err = self.forward(Tensor(X_te), Tensor(y_te)).realize_cached_data()
        accuracy = 1 - test_err
        return accuracy

if __name__ == "__main__":
    (X_tr, y_tr), (X_te, y_te) = parse_mnist()
    model = CNN()
    model.train_nn(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.001, batch=100, beta1=0.9, beta2=0.999, using_adam=False)  
    accuracy = model.test_nn(X_te, y_te)
    print(f"Accuracy after train overall: {accuracy * 100:.2f}%")