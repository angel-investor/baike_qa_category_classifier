import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from config import Config

# 加载配置
conf = Config()


# 定义bert模型
class BertClassifier(nn.Module):
    def __init__(self):
        # 初始化父类类的构造函数
        super().__init__()
        # 下面的BertModel是从transformers库中加载的预训练模型
        # config.bert_path是预训练模型的路径
        self.bert = BertModel.from_pretrained(conf.bert_path)
        # 定义随机失活
        self.dropout = nn.Dropout(0.3)
        # 定义全连接层（fc），用于分类任务
        # 输入尺寸是Bert模型隐藏层的大小，即768（对于Base模型）
        # 输出尺寸是类别数量10，由config.num_classes指定
        self.fc = nn.Linear(conf.bert_config.hidden_size, conf.num_classes)

    def forward(self, input_ids, attention_mask):
        # 使用BERT模型处理输入的token ID和注意力掩码，获取BERT模型的输出
        # outputs拆包是: _,pooled池化
        outputs = self.bert(
            input_ids=input_ids,  # 输入的token ID
            attention_mask=attention_mask,  # 注意力掩码用于区分有效token和填充token
            return_dict=True    # 让输出变为字典形式，更直观安全
        )
        # 获取池化层输出
        pooler_output = outputs.pooler_output
        # 使用随机失活层对池化层输出进行随机失活
        pooler_output = self.dropout(pooler_output)
        # print('outputs-->',outputs)  # 观察结果
        # 通过全连接层对BERT模型的输出进行分类
        logits = self.fc(pooler_output)
        # 返回分类的logits（未归一化的预测分数）
        return logits


# 测试以上模型
def test_bert_classifier():
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(conf.bert_path)

    # 示例文本
    texts = ["王者荣耀", "今天天气真好"]

    # 编码文本
    encoded_inputs = tokenizer(
        texts,
        # padding=True,  #  所有的填充到文本最大长度
        padding="max_length",  # 所有的填充到指定的max_length长度
        max_length=256,
        truncation=True,  # 如果超出指定的max_length长度，则截断
        return_tensors="pt"  # 返回 pytorch 张量,"pt" 时，分词器会将输入文本转换为模型可接受的格式
    )

    # 获取 input_ids 和 attention_mask
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    print('input_ids-->', input_ids)
    print('attention_mask-->', attention_mask)

    # 创建自定义的bert模型
    model = BertClassifier()  # __init__()执行了

    # 预测
    logits = model(input_ids=input_ids, attention_mask=attention_mask)  # forward()执行了
    print('logits-->', logits)  # 每一行对应一个样本，每个数字表示该样本属于某一类别的“得分”（logit），没有经过 softmax 归一化。

    # 获取预测概率
    probs = torch.softmax(logits, dim=-1)
    print('probs-->', probs)  # 归一化后该样本属于某类的概率（范围在 0~1 之间）,概率最高的就是预测结果

    # 获取预测结果
    preds = torch.argmax(logits, dim=-1)
    print('preds-->', preds)  # 得到每个样本的预测类别。表示两个输入文本被模型预测为类别 6（从 0 开始计数）。


if __name__ == '__main__':
    test_bert_classifier()