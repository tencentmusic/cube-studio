import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import pickle

# 读取数据
print("正在读取数据...")
data = pd.read_csv('data.csv')  # 你的CSV文件

# 检查数据
print("数据形状:", data.shape)
print("数据列名:", data.columns.tolist())
print("\n前5行数据:")
print(data.head())

# 分离特征和目标变量
X = data.drop('y', axis=1)  # 特征
y = data['y']  # 目标变量

# 确保数据类型一致性
X = X.astype(np.float32)  # 将所有特征转换为float32
y = y.astype(np.int32)    # 将目标变量转换为int32

print(f"\n特征数据类型: {X.dtypes}")
print(f"目标变量数据类型: {y.dtype}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 创建并训练决策树模型
print("\n正在训练决策树模型...")
model = DecisionTreeClassifier(
    max_depth=5,           # 限制树的最大深度，防止过拟合
    min_samples_split=10,  # 内部节点再划分所需最小样本数
    min_samples_leaf=5,    # 叶节点最少样本数
    random_state=42
)

model.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"模型准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 方法1: 使用joblib保存模型（推荐）
print("\n使用joblib保存模型...")
model_filename = 'decision_tree_model.pkl'
joblib.dump(model, model_filename, compress=3)
print(f"模型已保存到: {model_filename}")

# 方法2: 同时保存为pickle格式（兼容性更好）
print("\n使用pickle保存模型...")
with open('decision_tree_model_pickle.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"模型已保存到: decision_tree_model_pickle.pkl")

# 保存特征列名（用于推理时验证输入）
feature_names = X.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print(f"特征名称已保存到: feature_names.json")

# 保存训练数据的统计信息（用于数据验证）
data_stats = {
    'feature_ranges': {
        col: {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'mean': float(X[col].mean()),
            'std': float(X[col].std()),
            'dtype': str(X[col].dtype)
        }
        for col in X.columns
    },
    'target_distribution': {
        'classes': y.unique().tolist(),
        'class_counts': y.value_counts().to_dict()
    }
}

with open('data_stats.json', 'w') as f:
    json.dump(data_stats, f, indent=2)
print(f"数据统计信息已保存到: data_stats.json")

# 保存模型配置信息
model_info = {
    'features': feature_names,
    'feature_dtypes': {col: str(X[col].dtype) for col in X.columns},
    'target_name': 'y',
    'model_type': 'decision_tree',
    'accuracy': float(accuracy),
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'model_parameters': model.get_params(),
    'classes': model.classes_.tolist(),
    'n_classes': len(model.classes_),
    'model_version': '1.0.0',
    'save_methods': ['joblib', 'pickle']
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"模型信息已保存到: model_info.json")

print("\n训练完成！")
print("=" * 50)
print("生成的文件:")
print("1. decision_tree_model.pkl - 主模型文件 (joblib格式)")
print("2. decision_tree_model_pickle.pkl - 备份模型文件 (pickle格式)")
print("3. feature_names.json - 特征名称")
print("4. data_stats.json - 数据统计信息")
print("5. model_info.json - 模型信息")