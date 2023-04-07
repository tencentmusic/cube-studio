# 加载相关的库
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 读取数据
df = pd.read_csv('data.csv')

# 处理数据
X = df.drop('y', axis=1) # 特征变量
y = df['y'] # 目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 划分训练集和测试集

# 定义逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)


#保存模型
joblib.dump(model, 'lr_model.pkl')
#加载模型
model = joblib.load('lr_model.pkl')


# 预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

