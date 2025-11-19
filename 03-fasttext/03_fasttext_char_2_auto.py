# 导入工具包
import fasttext
from config import Config
import datetime

"""
fastText 自从 v0.9.0 版本起，支持了内置的 自动调参（Auto Tuning） 功能，
通过设置 autotuneValidationFile 参数，可以让模型在训练时根据你提供的验证集自动调整以下超参数：
学习率（lr）
epoch 数量（epoch）
词向量维度（dim）
字符 n-gram 长度范围（minn, maxn）
单词 n-gram 数量（wordNgrams）
损失函数类型（loss）
注意: 若你没有设置 autotuneValidationFile，fastText 将不会进行自动调参!!!
"""
# 加载配置
conf = Config()

# 获取时间
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")

# 自动调参学习
# 1、模型训练
model = fasttext.train_supervised(input=conf.process_train_datapath_char,
                                  autotuneValidationFile=conf.process_dev_datapath_char,
                                  autotuneDuration=300,  # 自动调参时长
                                  thread=8, verbose=3,  # 可设置0,1,2,3,其中3输出更详细的调试信息，包括自动调参过程、每个 epoch 的 loss等
                                  seed=42  # 设置随机数生成器的种子值，确保实验的可重复性
)
# 2、模型保存
# 保存路径
model_path = conf.ft_model_save_path + f"/model_char_2_auto_{current_time}.bin"
model.save_model(model_path)

# 3、模型预测
print('模型预测-->', model.predict("嘴 群 里 有 许 多 硬 结 是 啥   嘴 群 里 有 许 多 硬 结 是 啥"))

# 4、模型词表的查看
print('模型词表的查看-->', model.words[:10])

# 5、模型子词查看
print('模型子词查看-->', model.get_subwords("硬"))
# 打印向量维度
print('打印向量维度-->', model.get_dimension())

# 6、模型评估
print('模型评估-->', model.test(conf.process_test_datapath_char))
