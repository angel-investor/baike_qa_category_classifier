import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from config import Config
import warnings
# 导入数据处理工具类
from a1_dataloader_utils import build_dataloader
# 导入bert模型
from a2_bert_classifer_model import BertClassifier

warnings.filterwarnings("ignore")

# 加载配置对象，包含模型参数、路径等
conf = Config()


def model2predict():
    # 准备数据
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    # 准备模型
    # 初始化bert分类模型
    model = BertClassifier()
    model.load_state_dict(torch.load(conf.model_save_path))
    # 将模型移动到指定的设备
    model.to(conf.device)
    # 设置模型为评估模式（禁用 dropout,并改变batch_norm行为）
    model.eval()
    # 初始化列表，存储预测结果和真实标签
    all_preds, all_labels = [], []
    # torch.no_grad()禁用梯度计算以提高效率并减少内存占用
    with torch.no_grad():
        # 4. 遍历数据加载器，逐批次进行预测
        for i, batch in enumerate(tqdm(test_dataloader, desc="测试集测试中...")):
            # 提取批次数据并移动到设备
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            labels = labels.to(conf.device)
            # 前向传播：模型预测
            outputs = model(input_ids, attention_mask=attention_mask)
            # print("output-->", outputs, outputs.shape)

            # 获取预测结果（最大 logits分数 对应的类别）
            y_pred_list = torch.argmax(outputs, dim=-1)

            # 存储预测和真实标签
            all_preds.extend(y_pred_list.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    f1score = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds)

    print("accuracy-->", accuracy)
    print("precision-->", precision)
    print("f1score-->", f1score)
    print("recall-->", recall)
    print("report-->", report)


if __name__ == '__main__':
    model2predict()