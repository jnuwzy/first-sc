import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子
seed_value = 2023
np.random.seed(seed_value)


# Sigmoid激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义逻辑回归算法
class LogisticRegression:
    def __init__(self, learning_rate=0.003, iterations=100):
        self.learning_rate = learning_rate  # 学习率
        self.iterations = iterations  # 迭代次数

    def fit(self, X, y):
        # 初始化参数
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0

        # 梯度下降
        for i in range(self.iterations):
            # 计算sigmoid函数的预测值, y_hat = w * x + b
            y_hat = sigmoid(np.dot(X, self.weights) + self.bias)

            # 计算损失函数
            loss = (-1 / len(X)) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

            # 计算梯度
            dw = (1 / len(X)) * np.dot(X.T, (y_hat - y))
            db = (1 / len(X)) * np.sum(y_hat - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 打印损失函数值
            if i % 10 == 0:
                print(f"Loss after iteration {i}: {loss}")

    # 预测
    def predict(self, X):
        y_hat = sigmoid(np.dot(X, self.weights) + self.bias)
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        return y_hat

    # 精度
    def score(self, y_pred, y):
        accuracy = (y_pred == y).sum() / len(y)
        return accuracy


# 导入数据
# 读取Excel文件
file_path = '../data/v2.xls'  # .. 表示上一级目录，即src目录
df = pd.read_excel(file_path)

# 分离特征和目标变量
X = df[['年龄', ' BMI ', '体脂百分比', '腰臀比', '肥胖度', '骨骼肌指数']]
y = df['胰岛素抵抗存在=1']


# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed_value)

# 训练模型
model = LogisticRegression(learning_rate=0.03, iterations=300)
model.fit(X_train, y_train)

# 结果
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

score_train = model.score(y_train_pred, y_train)
score_test = model.score(y_test_pred, y_test)

print('训练集Accuracy: ', score_train)
print('测试集Accuracy: ', score_test)

# 可视化决策边界
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()
