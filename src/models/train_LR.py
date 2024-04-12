import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import stats

# 读取Excel文件
file_path = '../data/v2.xls'
data = pd.read_excel(file_path)

# 分离特征和目标变量
X = data[['年龄', ' BMI ', '体脂百分比', '腰臀比', '肥胖度', '骨骼肌指数']]
y = data['胰岛素抵抗存在=1']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 StandardScaler 对象
scaler = StandardScaler()

# 对训练数据和测试数据进行标准化处理
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建一个可调参的逻辑回归模型实例
logreg = LogisticRegression(
    penalty='l2',       # 正则化项，'l1'或'l2'，默认为'l2'
    C=1.0,             # 正则化强度的倒数，较小的值指定更强的正则化
    solver='sag',    # 优化算法，{'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, 默认为'lbfgs'
    max_iter=1000,      # 最大迭代次数
    random_state=42,   # 随机数生成器的种子，用于可重复性
    tol=1e-4,          # 优化算法的收敛阈值
    verbose=0,         # 对于liblinear和lbfgs求解器，设置verbosity级别
    warm_start=False,  # 是否使用前一次训练的解作为初始化，适合在`solver='lbfgs', 'sag', 'saga'`时使用
    n_jobs=None,      # 用于计算的CPU核数，-1表示使用所有可用的核
    l1_ratio=None     # 'l1'正则化与'l2'正则化之间的弹性网混合比例
)
# 训练模型
logreg.fit(X_train_scaled, y_train)

# 预测测试集结果
y_pred = logreg.predict(X_test_scaled)

# 获取唯一的类别
unique_classes = np.unique(np.concatenate((y_test, y_pred)))

# 计算整体的准确率
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.2f}".format(accuracy))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 查看每个特征的系数
coef = logreg.coef_[0]
feature_names = X.columns

# 计算P值
_, p_values = stats.ttest_ind(X_train_scaled[y_train == 1], X_train_scaled[y_train == 0])

# 设置matplotlib以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 绘制特征和系数的关系图
plt.figure(figsize=(12, 6))

# 绘制系数
plt.subplot(1, 2, 1)
plt.bar(feature_names, coef)
plt.xticks(rotation=45)
plt.ylabel('系数')
plt.title('逻辑回归的系数')

# 绘制P值
plt.subplot(1, 2, 2)
plt.bar(feature_names, -np.log10(p_values))  # 对P值取对数处理以便于可视化
plt.xticks(rotation=45)
plt.ylabel('-log10(P值)')
plt.title('逻辑回归的P值')

plt.tight_layout()
plt.show()
