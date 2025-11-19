from flask import Flask, request, jsonify
from ft_predict_func import predict_func
import warnings

warnings.filterwarnings("ignore")


app = Flask(__name__)

# 创建API接口
@app.route('/predict', methods=['POST'])
def predict():
    # 获取json数据
    data = request.get_json()
    print(type(data))
    print(data)
    # 预测结果
    result = predict_func(data)
    return jsonify(result)


if __name__ == '__main__':
    # 启动服务
    app.run(host='127.0.0.1', port=8009, debug=True)
