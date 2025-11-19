import time
import datetime
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from config import Config

conf = Config()

def model2train():
    # ======================
    # 1. 加载数据
    # ======================
    print("正在加载数据集...")
    df_data = pd.read_csv(conf.process_train_datapath, sep='\t')
    print("数据量：", len(df_data))

    words = df_data["words"].astype(str)
    labels = df_data["label"].astype(str)

    # 加载停用词
    stop_words = [w.strip() for w in open(conf.stop_words_path, encoding='utf-8')]

    # ======================
    # 2. TF-IDF 特征提取优化
    # ======================
    print("开始进行 TF-IDF 特征提取...")

    tfidf = TfidfVectorizer(
        stop_words=stop_words,
        tokenizer=str.split,          # 避免英文拆分问题
        token_pattern=None,
        max_features=30000            # 有效避免内存爆炸
    )

    features = tfidf.fit_transform(words)
    print("特征维度：", features.shape)

    # ======================
    # 3. 划分训练测试集
    # ======================
    print("划分训练集与测试集...")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=0.2,
        random_state=28,
        stratify=labels               # 保留类别分布，更适合你的多类别任务
    )

    # ======================
    # 4. 随机森林训练优化
    # ======================
    print("开始训练模型...")
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=25,
        min_samples_split=4,
        n_jobs=-1,
        class_weight="balanced_subsample"   # 类别不均衡数据的利器
    )

    with tqdm(total=1, desc="随机森林训练进度") as pbar:
        rf.fit(x_train, y_train)
        pbar.update(1)

    # ======================
    # 5. 模型预测（向量化）
    # ======================
    print("开始模型预测和评估...")
    y_pred = rf.predict(x_test)   # ！！一次性预测，比你之前快 100 倍

    # ======================
    # 6. 模型评估
    # ======================
    print("\n模型评估结果如下:")
    print("准确率：", accuracy_score(y_test, y_pred))
    print("精确率：", precision_score(y_test, y_pred, average='macro'))
    print("召回率：", recall_score(y_test, y_pred, average='macro'))
    print("F1值：", f1_score(y_test, y_pred, average='macro'))
    print("\n详细分类报告：\n", classification_report(y_test, y_pred))

    # ======================
    # 7. 保存模型（使用 joblib）
    # ======================
    print("开始保存模型和向量化器...")

    joblib.dump(rf, conf.rf_model_save_path)
    joblib.dump(tfidf, conf.tfidf_model_save_path)

    print("模型与向量化器已保存")


if __name__ == "__main__":
    start_time = time.time()
    model2train()
    end_time = time.time()

    cost = end_time - start_time
    print("模型训练时长：", str(datetime.timedelta(seconds=cost)))
