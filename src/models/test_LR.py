import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 假设你有一个CSV文件，其中包含以下列：年龄、BMI、体脂百分比、腰臀比、肥胖度、骨骼肌指数、胰岛素抵抗指数
# 胰岛素抵抗指数已经用1和0标记好了
# 读取Excel文件
file_path = '../data/v2.xls'  # .. 表示上一级目录，即src目录
data = pd.read_excel(file_path)

# 查看数据的前几行
print(data.head())

# 分离特征和目标变量
X = data[['年龄', ' BMI ', '体脂百分比', '腰臀比', '肥胖度', '骨骼肌指数']]
y = data['胰岛素抵抗存在=1']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型实例
logreg = LogisticRegression()

# 训练模型
logreg.fit(X_train, y_train)

# 预测测试集结果
y_pred = logreg.predict(X_test)

# 查看模型评估结果
print(classification_report(y_test, y_pred))

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