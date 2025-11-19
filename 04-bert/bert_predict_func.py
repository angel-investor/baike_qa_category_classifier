import torch
from a2_bert_classifer_model import BertClassifier
from config import Config
from a1_dataloader_utils import build_dataloader

# 加载配置
conf = Config()

# 准备模型
model = BertClassifier()
# 加载模型参数
model.load_state_dict(torch.load(conf.model_save_path))
# 添加模型到指定设备
model.to(conf.device)
# 设置模型为评估模式
model.eval()


# 定义predict_fun函数预测函数
def predict_func(data_dict):
    """
    根据用户录入数据,返回分类信息
    :param 参数 data_dict: {"text":"状元心经：考前一周重点是回顾和整理"}
    :return: 返回 data_dict: {"text":"状元心经：考前一周重点是回顾和整理", "pred_class":"education"}
    """
    # 获取文本
    text = data_dict['text']
    # 将文本转为id
    text_tokens = conf.tokenizer.batch_encode_plus([text],
                                                   padding="max_length",
                                                   max_length=conf.pad_size)
    # print("text_tokens-->", text_tokens)
    # 获取input_ids和attention_mask
    input_ids = text_tokens['input_ids']
    attention_mask = text_tokens['attention_mask']
    # 将input_ids和attention_mask转为tensor, 并指定到设备
    input_ids = torch.tensor(input_ids).to(conf.device)
    attention_mask = torch.tensor(attention_mask).to(conf.device)

    # 设置不进行梯度计算(在该上下文中禁用梯度计算，提升推理速度并减少内存占用)
    with torch.no_grad():
        # 前向传播(模型预测)
        logits = model(input_ids, attention_mask)
        # print("logits-->", logits, logits.shape)
        # 获取预测类别索引张量
        preds = torch.argmax(logits, dim=-1)
        # print("preds-->", preds)
        # 获取预测类别索引标量
        pred_idx = preds.item()
        # 获取类别名称
        pred_class = conf.class_list[pred_idx]
        # print('pred_class-->', pred_class)
        # 将预测结果添加到data_dict中
        data_dict['pred_class'] = pred_class
    # 返回data_dict
    return data_dict


if __name__ == '__main__':
    data_dict = {'text': '长安志翔混合动力能单加发电机吗？ 长安志翔混合动力能单加吗？'}
    result = predict_func(data_dict)
    print("result-->", result)
