import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# accuracy:0.69

# 假设你有一个CSV文件，其中包含以下列：年龄、BMI、体脂百分比、腰臀比、肥胖度、骨骼肌指数、胰岛素抵抗指数
# 胰岛素抵抗指数已经用1和0标记好了
# 读取Excel文件
file_path = '../data/v2.xls'  # .. 表示上一级目录，即src目录
data = pd.read_excel(file_path)

# 查看数据的前几行
#print(data.head())

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

# 查看模型评估结果
print(classification_report(y_test, y_pred))

# 获取唯一的类别
unique_classes = np.unique(np.concatenate((y_test, y_pred)))

# 计算整体的准确率
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.2f}".format(accuracy))
print("\nClassification Report:")
for class_name in unique_classes:
    report_class = classification_report(y_test, y_pred, labels=[class_name], output_dict=True)
    print("Class: {:<10}".format(class_name))
    print("Precision: {:.2f}".format(report_class[str(class_name)]['precision']))
    print("Recall: {:.2f}".format(report_class[str(class_name)]['recall']))
    print("F1-score: {:.2f}".format(report_class[str(class_name)]['f1-score']))
    print("Support: {:<5}".format(report_class[str(class_name)]['support']))
    print("\n")

# 查看每个特征的系数，这可以解释它们对胰岛素抵抗的影响
coef = logreg.coef_[0]
feature_names = X.columns

# 设置matplotlib以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 绘制特征和系数的关系图
plt.figure(figsize=(10, 6))
plt.bar(feature_names, coef)
plt.xticks(rotation=45)  # 旋转x轴标签以避免重叠
plt.ylabel('系数')  # 设置y轴标签为中文
plt.title('逻辑回归的系数')  # 设置图表标题为中文
plt.show()

# 注意：系数的正负可以解释特征与胰岛素抵抗的关系方向，但系数的绝对值大小需要谨慎解释，因为它还受到特征缩放的影响。