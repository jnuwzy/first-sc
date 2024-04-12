import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 使用相对路径读取data文件夹下的v2.xls文件
file_path = '../data/v2.xls'
df = pd.read_excel(file_path)

# 分离特征和目标变量
X = df[['年龄', ' BMI ', '体脂百分比', '腰臀比', '肥胖度', '骨骼肌指数']]
y = df['胰岛素抵抗存在=1']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 StandardScaler 对象
scaler = StandardScaler()

# 对训练数据和测试数据进行标准化处理
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 创建SVM分类器实例
clf = svm.SVC(kernel='linear')  # 线性核，您也可以尝试其他核函数，如'rbf'

# 训练SVM分类器
clf.fit(X_train_scaled, y_train)

# 使用训练好的模型进行预测
y_pred = clf.predict(X_test_scaled)

# 计算并打印预测的准确率
accuracy = accuracy_score(y_test, y_pred)
print("SVM分类器的准确率: {:.2f}%".format(accuracy * 100))