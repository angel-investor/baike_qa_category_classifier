import torch
import datetime
from transformers import BertModel, BertTokenizer, BertConfig

# 获取当前日期字符串
current_date = datetime.datetime.now().date().strftime("%Y%m%d")


# 配置类
class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        self.model_name = "bert"  # 模型名称
        # 路径
        # 根目录
        self.root_path = r'D:\project\workspace\baike_qa_category_classifier/'
        # 原始数据路径
        self.train_datapath = self.root_path + 'data/train.txt'
        self.test_datapath = self.root_path + 'data/test.txt'
        self.dev_datapath = self.root_path + 'data/valid.txt'
        # 类别文档
        self.class_path = self.root_path + "data/category_list.txt"
        # 类别名列表
        self.class_list = [line.strip() for line in open(self.class_path, encoding="utf-8")]  # 类别名单

        # 模型训练保存路径
        self.model_save_path = self.root_path + "04-bert/save_models/bert_classifer_model.pt"  # 模型训练结果保存路径

        # 模型训练+预测的时候  训练设备，如果GPU可用，则为cuda，否则为cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bert_path = self.root_path + "/04-bert/bert-base-chinese"  # 预训练BERT模型的路径
        self.bert_model = BertModel.from_pretrained(self.bert_path)  # 加载预训练BERT模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path)  # BERT模型的配置

        # 参数
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 5  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 256 # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率

        # 量化模型存放地址
        # 注意: 量化的时候模型需要的设备首选是cpu
        self.bert_model_quantization_model_path = self.root_path + "04-bert/save_models/bert_classifer_quantization_model.pt"  # 模型训练结果保存路径


if __name__ == '__main__':
    # 测试
    conf = Config()
    print('conf.device-->', conf.device)
    print('conf.class_list-->', conf.class_list)
    print('conf.tokenizer-->', conf.tokenizer)
    input_size = conf.tokenizer.convert_tokens_to_ids(["你", "好", "中", "人"])
    print('input_size-->', input_size)
    # print('conf.bert_model-->', conf.bert_model)
    # print('conf.bert_config-->', conf.bert_config)
