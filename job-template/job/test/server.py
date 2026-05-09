from flask import Flask, request, jsonify
import joblib
import pickle
import pandas as pd
import numpy as np
import json
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)  # 允许跨域请求


class ModelLoader:
    """模型加载器，处理不同格式的模型文件"""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_info = None
        self.data_stats = None
        self.load_models()

    def load_models(self):
        """尝试加载模型文件，优先使用joblib，失败则使用pickle"""
        try:
            print("尝试使用joblib加载模型...")
            self.model = joblib.load('decision_tree_model.pkl')
            print("✓ 使用joblib加载模型成功")
        except Exception as e:
            print(f"joblib加载失败: {e}")
            print("尝试使用pickle加载模型...")
            try:
                with open('decision_tree_model_pickle.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                print("✓ 使用pickle加载模型成功")
            except Exception as e2:
                print(f"pickle加载失败: {e2}")
                raise RuntimeError("所有模型加载方法都失败")

        # 加载其他必要文件
        with open('feature_names.json', 'r') as f:
            self.feature_names = json.load(f)

        with open('model_info.json', 'r') as f:
            self.model_info = json.load(f)

        try:
            with open('data_stats.json', 'r') as f:
                self.data_stats = json.load(f)
        except:
            self.data_stats = None

        print(f"模型加载完成:")
        print(f"  - 特征数量: {len(self.feature_names)}")
        print(f"  - 模型类别: {self.model_info.get('model_type')}")
        print(f"  - 准确率: {self.model_info.get('accuracy'):.4f}")

    def validate_input(self, data):
        """验证输入数据"""
        errors = []

        # 检查特征数量
        if len(data) != len(self.feature_names):
            errors.append(f"特征数量不正确，期望{len(self.feature_names)}，收到{len(data)}")

        # 检查特征名称
        missing_features = [f for f in self.feature_names if f not in data]
        if missing_features:
            errors.append(f"缺少特征: {missing_features}")

        # 检查数据类型
        for feature in self.feature_names:
            if feature in data:
                try:
                    # 尝试转换为float
                    float(data[feature])
                except (ValueError, TypeError):
                    errors.append(f"特征 '{feature}' 的值 '{data[feature]}' 不是有效的数字")

        return errors

    def preprocess_input(self, data):
        """预处理输入数据"""
        # 确保正确的特征顺序和数据类型
        processed_data = {}
        for feature in self.feature_names:
            if feature in data:
                try:
                    # 转换为float32以匹配训练时的数据类型
                    processed_data[feature] = np.float32(data[feature])
                except:
                    processed_data[feature] = np.float32(0)  # 如果转换失败，使用默认值
            else:
                processed_data[feature] = np.float32(0)  # 缺失值用0填充

        return processed_data


# 初始化模型加载器
print("正在初始化模型...")
try:
    model_loader = ModelLoader()
    print("✓ 模型初始化成功")
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    traceback.print_exc()
    model_loader = None


@app.route('/')
def home():
    """API首页"""
    if model_loader is None:
        return jsonify({'error': '模型加载失败'}), 500

    return jsonify({
        'message': '决策树模型推理API',
        'status': 'running',
        'model_info': {
            'model_type': model_loader.model_info.get('model_type'),
            'accuracy': model_loader.model_info.get('accuracy'),
            'features_count': len(model_loader.feature_names),
            'classes': model_loader.model_info.get('classes'),
            'version': model_loader.model_info.get('model_version', '1.0.0')
        },
        'endpoints': {
            'GET /': 'API首页',
            'POST /predict': '单个样本预测',
            'POST /batch_predict': '批量预测',
            'GET /model_info': '获取详细模型信息',
            'GET /health': '健康检查'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    if model_loader is None:
        return jsonify({'status': 'unhealthy', 'error': '模型未加载'}), 500

    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': pd.Timestamp.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """单个样本预测"""
    if model_loader is None:
        return jsonify({'error': '模型未加载'}), 500

    try:
        # 获取JSON数据
        data = request.get_json()

        if not data:
            return jsonify({'error': '没有提供数据'}), 400

        # 验证输入
        validation_errors = model_loader.validate_input(data)
        if validation_errors:
            return jsonify({
                'error': '输入验证失败',
                'details': validation_errors,
                'required_features': model_loader.feature_names
            }), 400

        # 预处理输入
        processed_data = model_loader.preprocess_input(data)

        # 创建DataFrame，确保特征顺序正确
        input_df = pd.DataFrame([processed_data], columns=model_loader.feature_names)

        # 确保数据类型与训练时一致
        for col in input_df.columns:
            input_df[col] = input_df[col].astype(np.float32)

        # 进行预测
        prediction = model_loader.model.predict(input_df)[0]

        # 获取预测概率
        try:
            prediction_proba = model_loader.model.predict_proba(input_df)[0].tolist()
        except:
            prediction_proba = None

        response = {
            'prediction': int(prediction),
            'class_label': model_loader.model_info['classes'][prediction]
            if prediction < len(model_loader.model_info['classes'])
            else 'unknown',
            'features_used': model_loader.feature_names,
            'input_values': {k: float(v) for k, v in processed_data.items()}
        }

        if prediction_proba:
            response['prediction_probability'] = prediction_proba
            response['confidence'] = float(max(prediction_proba))

        return jsonify(response)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"预测错误: {str(e)}")
        print(f"错误详情: {error_trace}")
        return jsonify({
            'error': '预测失败',
            'message': str(e),
            'traceback': error_trace if app.debug else None
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测"""
    if model_loader is None:
        return jsonify({'error': '模型未加载'}), 500

    try:
        data = request.get_json()

        if not data or 'samples' not in data:
            return jsonify({'error': '请提供samples数组'}), 400

        samples = data['samples']

        if not isinstance(samples, list):
            return jsonify({'error': 'samples应该是一个数组'}), 400

        if len(samples) > 1000:  # 限制批量大小
            return jsonify({'error': '批量预测最多支持1000个样本'}), 400

        results = []
        errors = []

        for i, sample in enumerate(samples):
            try:
                # 验证输入
                validation_errors = model_loader.validate_input(sample)
                if validation_errors:
                    errors.append({
                        'sample_index': i,
                        'error': '输入验证失败',
                        'details': validation_errors
                    })
                    continue

                # 预处理输入
                processed_data = model_loader.preprocess_input(sample)

                # 创建DataFrame
                input_df = pd.DataFrame([processed_data], columns=model_loader.feature_names)

                # 确保数据类型
                for col in input_df.columns:
                    input_df[col] = input_df[col].astype(np.float32)

                # 进行预测
                prediction = model_loader.model.predict(input_df)[0]

                # 获取预测概率
                try:
                    prediction_proba = model_loader.model.predict_proba(input_df)[0].tolist()
                    confidence = float(max(prediction_proba))
                except:
                    prediction_proba = None
                    confidence = None

                results.append({
                    'sample_index': i,
                    'prediction': int(prediction),
                    'class_label': model_loader.model_info['classes'][prediction]
                    if prediction < len(model_loader.model_info['classes'])
                    else 'unknown',
                    'confidence': confidence,
                    'prediction_probability': prediction_proba
                })

            except Exception as e:
                errors.append({
                    'sample_index': i,
                    'error': str(e)
                })

        return jsonify({
            'success_count': len(results),
            'error_count': len(errors),
            'results': results,
            'errors': errors if errors else None
        })

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"批量预测错误: {str(e)}")
        return jsonify({
            'error': '批量预测失败',
            'message': str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    if model_loader is None:
        return jsonify({'error': '模型未加载'}), 500

    return jsonify(model_loader.model_info)


if __name__ == '__main__':
    # 设置debug模式（生产环境应设为False）
    debug_mode = True

    if model_loader is None:
        print("警告: 模型加载失败，API可能无法正常工作")

    print(f"\n启动Flask API服务...")
    print(f"访问 http://localhost:5000 查看API文档")
    print(f"Debug模式: {'开启' if debug_mode else '关闭'}")

    app.run(host='0.0.0.0', port=5000, debug=debug_mode)