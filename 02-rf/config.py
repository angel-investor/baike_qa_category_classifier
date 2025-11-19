class Config:
    def __init__(self):
        self.root_path = r'D:\project\workspace\baike_qa_category_classifier/'
        # 原始数据路径
        self.train_datapath = self.root_path + 'data/final_train.json'
        self.test_datapath = self.root_path + 'data/final_test.json'
        self.dev_datapath = self.root_path + 'data/final_valid.json'
        # 类别文档
        self.class_doc_path = self.root_path + "data/category_list.txt"
        # 停用词路径
        self.stop_words_path = self.root_path + "data/stopwords.txt"

        # 处理后的数据路径
        self.process_train_datapath = self.root_path + "02-rf/final_data/train_process.txt"
        self.process_test_datapath = self.root_path + "02-rf/final_data/test_process.txt"
        self.process_dev_datapath = self.root_path + "02-rf/final_data/dev_process.txt"
        self.process_dev_datapath_simple = self.root_path + "02-rf/final_data/test_process_simple.txt"


        # 保存模型路径
        self.rf_model_save_path = self.root_path + r"02-rf/save_model/rf_model.pkl"
        self.tfidf_model_save_path = self.root_path + r"02-rf/save_model/tfidf_model.pkl"
        # 模型预测结果
        self.model_predict_result = self.root_path + r"02-rf/result/predict_result.txt"


if __name__ == '__main__':
    conf = Config()
    print('conf-->', conf.train_datapath)