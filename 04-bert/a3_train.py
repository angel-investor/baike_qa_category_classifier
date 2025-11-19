import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from model2dev_utils import model2dev
from config import Config
import warnings
# 导入数据处理工具类
from a1_dataloader_utils import build_dataloader
# 导入bert模型
from a2_bert_classifer_model import BertClassifier

warnings.filterwarnings("ignore")

# 加载配置对象，包含模型参数、路径等
conf = Config()


def model2train():
    """
    训练 BERT 分类模型并在验证集上评估，保存最佳模型。
    参数：无显式参数，所有配置通过全局 conf 对象获取。
    返回：无返回值，训练过程中保存最佳模型到指定路径。
    """
    # 准备数据
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()

    # 准备模型
    # 初始化bert分类模型
    model = BertClassifier()
    # 将模型移动到指定的设备
    model.to(conf.device)

    # 准备损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 准备优化器
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)

    # 开始训练模型
    # 初始化F1分数，用于保存最好的模型
    best_f1 = 0.0
    # 外层循环遍历每个训练轮次
    #  （每次需要设置训练模式，累计损失，预存训练集测试和真实标签）
    for epoch in range(conf.num_epochs):
        model.train()

        # ====== epoch 级变量 ======
        epoch_loss = 0.0
        epoch_preds = []
        epoch_labels = []

        # ====== batch 级变量 ======
        batch_loss = 0.0
        batch_preds = []
        batch_labels = []

        for i, batch in enumerate(tqdm(train_dataloader, desc=f"训练集训练中，轮次 {epoch + 1}")):

            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            labels = labels.to(conf.device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            y_pred_list = torch.argmax(logits, dim=1)

            # ==== 更新 epoch 级变量 ====
            epoch_loss += loss.item()
            epoch_preds.extend(y_pred_list.cpu().tolist())
            epoch_labels.extend(labels.cpu().tolist())

            # ==== 更新 batch 级变量（用于打印） ====
            batch_loss += loss.item()
            batch_preds.extend(y_pred_list.cpu().tolist())
            batch_labels.extend(labels.cpu().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===== 每 10 个 batch 打印训练信息 =====
            if (i + 1) % 10 == 0:
                acc = accuracy_score(batch_labels, batch_preds)
                f1 = f1_score(batch_labels, batch_preds, average='macro')
                avg_loss = batch_loss / len(batch_preds)

                print(f"\n轮次: {epoch + 1}, 批次: {i + 1}, "
                      f"损失: {avg_loss:.4f}, acc:{acc:.4f}, f1:{f1:.4f}")

                # 清空 batch 累计
                batch_loss = 0.0
                batch_preds = []
                batch_labels = []

        # ========== epoch 结束，计算 epoch 级指标 ==========
        epoch_avg_loss = epoch_loss / len(train_dataloader)
        epoch_acc = accuracy_score(epoch_labels, epoch_preds)
        epoch_f1 = f1_score(epoch_labels, epoch_preds, average='macro')

        print(f"\n轮次 {epoch + 1} 训练集：损失:{epoch_avg_loss:.4f}, "
              f"acc:{epoch_acc:.4f}, f1:{epoch_f1:.4f}")


        # ===== 验证集评估 =====
        model.eval()
        report, f1score, accuracy, precision, recall = model2dev(model, dev_dataloader, conf.device)
        print(f"轮次 {epoch+1} 验证集评估报告:\n{report}")
        print(f"验证集 f1:{f1score:.4f}, accuracy:{accuracy:.4f}, precision:{precision:.4f}, recall:{recall:.4f}")

        # 保存最佳模型
        if f1score > best_f1:
            best_f1 = f1score
            torch.save(model.state_dict(), conf.model_save_path)
            print(f"保存模型成功，最佳 f1:{best_f1:.4f}")

if __name__ == '__main__':
    model2train()
