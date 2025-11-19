import jieba
import pickle
from config import Config
import warnings
warnings.filterwarnings('ignore')

# 加载配置
conf = Config()
# 忽略警告信息

# 加载模型
with open(conf.rf_model_save_path, 'rb') as f:
    model = pickle.load(f)
# 加载向量化器
with open(conf.tfidf_model_save_path, 'rb') as f:
    tfidf = pickle.load(f)


def predict_fun(data):
    # jieba分词
    words = " ".join(jieba.lcut(data['text'])[:30])
    # 向量化
    features = tfidf.transform([words])
    # 预测
    y_pred = model.predict(features)[0]  # 模型直接输出类别字符串
    data["pred_class"] = y_pred
    return data


if __name__ == '__main__':
    data = {"text": "求香辣鱼的做法？求香辣鱼的具体做法 求香辣的具体做法"}
    result = predict_fun(data)
    print(result)
