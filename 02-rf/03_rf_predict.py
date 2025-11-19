import pandas as pd
import pickle
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from config import Config
# 加载配置
conf = Config()
# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')


def model2predict():

    # 加载模型和向量化器
    # rf = load(conf.rf_model_save_path)
    # tfidf = load(conf.tfidf_model_save_path)
    with open(conf.rf_model_save_path, 'rb') as f:
        rf = pickle.load(f)
    with open(conf.tfidf_model_save_path, 'rb') as f:
        tfidf = pickle.load(f)

    # 读取dev数据集
    df_data = pd.read_csv(conf.process_test_datapath, sep='\t')
    words = df_data["words"]

    # 通过tf-idf进行向量化
    features = tfidf.transform(words)

    # 预测
    y_pred = rf.predict(features)
    # 打印准确率，精确率，召回率，F1值
    print("准确率：", accuracy_score(df_data["label"], y_pred))
    print("精确率：", precision_score(df_data["label"], y_pred, average='macro'))
    print("召回率：", recall_score(df_data["label"], y_pred, average='macro'))
    print("F1值：", f1_score(df_data["label"], y_pred, average='macro'))
    # 打印评估报告
    print('评估报告-->', classification_report(df_data["label"], y_pred))

    # 保存结果
    df_data["pred_label"] = y_pred
    df_data.to_csv(conf.model_predict_result, sep='\t', index=False)


if __name__ == '__main__':
    model2predict()