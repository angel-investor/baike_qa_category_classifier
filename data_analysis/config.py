class Config:
    def __init__(self):
        self.root_path = r'D:\project\workspace\baike_qa_category_classifier/'
        # 原始数据集
        self.train_path = self.root_path + 'data/baike_qa_train.json'
        self.valid_path = self.root_path + 'data/baike_qa_valid.json'
        self.test_path = self.root_path + 'data/baike_qa_test.json'


if __name__ == '__main__':
    # 测试配置文件
    conf = Config()
    # 打印参数路径
    print(conf.train_path)
    print(conf.valid_path)
    print(conf.test_path)
