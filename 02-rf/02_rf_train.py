import time
import datetime
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from config import Config

# 配置对象
conf = Config()


def model2train():
    # 读取训练数据集
    print("正在加载数据集...")
    df_data = pd.read_csv(conf.process_train_datapath, sep='\t')
    print(len(df_data))

    words = df_data["words"]
    labels = df_data["label"]

    # 读取停用词
    stop_words = [line.strip() for line in open(conf.stop_words_path, encoding='utf-8').readlines()]
    print(stop_words)
    # 创建 TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words=stop_words)
    # 将文本转为词频矩阵
    print("开始进行 TF-IDF 特征提取...")
    features = tfidf.fit_transform(words)

    # 划分训练集和测试集
    print("划分训练集与测试集...")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=28)

    # 训练模型
    rf = RandomForestClassifier(n_estimators=120,
                                max_depth=25,
                                min_samples_split=4,
                                n_jobs=-1,
                                class_weight="balanced_subsample"   # 类别不均衡数据的利器
                                )
    print("开始训练模型...")
    with tqdm(total=1, desc="随机森林训练进度") as pbar:
        rf.fit(x_train, y_train)
        pbar.update(1)

    # 模型预测
    print("开始模型预测和评估...")
    y_pred = []

    with tqdm(total=x_test.shape[0], desc="预测进度") as pbar:
        for i in range(x_test.shape[0]):
            y_pred.append(rf.predict(x_test[i]))
            pbar.update(1)

    y_pred = [p[0] for p in y_pred]

    # 模型评估
    print("\n模型评估结果如下:")
    print("准确率：", accuracy_score(y_test, y_pred))
    print("精确率：", precision_score(y_test, y_pred, average='macro'))
    print("召回率：", recall_score(y_test, y_pred, average='macro'))
    print("F1值：", f1_score(y_test, y_pred, average='macro'))
    print('评估报告-->', classification_report(y_test, y_pred))

    print("开始保存模型和向量化器...")
    # 保存模型
    with open(conf.rf_model_save_path, 'wb') as f:
        pickle.dump(rf, f)
    print("模型已保存")

    # 保存向量化器
    with open(conf.tfidf_model_save_path, 'wb') as f:
        pickle.dump(tfidf, f)
    print("向量化器已保存")


if __name__ == '__main__':
    start_time = time.time()
    model2train()
    end_time = time.time()

    cost = end_time - start_time
    print("模型训练时长：", str(datetime.timedelta(seconds=cost)))
