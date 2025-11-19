from flask import Flask, request, jsonify
from rf_predict_func import predict_fun
import warnings
warnings.filterwarnings('ignore')

# 创建app
app = Flask(__name__)


# 创建预测接口
# 用户访问url: http://127.0.0.1:8008/predict
@app.route('/predict', methods=['POST'])
def predict():
    # 获取post请求携带的用户数据
    data = request.get_json()
    print('data-->', data)
    print('type(data)-->', type(data))
    # 根据用户请求数据进行预测
    result = predict_fun(data)
    print("result-->", result)
    # 返回结果给用户
    print("Response-->", jsonify(result))
    return jsonify(result)


# 启动服务，测试
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8008, debug=True)
