# 导入工具包
import fasttext
from config import Config


# 获取时间
conf = Config()

# 1、模型训练
model = fasttext.train_supervised(input=conf.process_train_datapath_word, seed=42)

# 2、模型保存
model_path = conf.ft_model_save_path + "/model_word_1_default.bin"
model.save_model(model_path)

# 3、模型预测
print('模型预测-->', model.predict("电脑 自动关机 是 怎么回事 不一会儿 就 会 自动关机   不一会儿 就 会 自动关机"))

# 4、模型词表的查看
print('模型词表的-->', model.words[:10])

# 5、模型子词查看
print('模型子词-->', model.get_subwords("电脑"))
# 输出模型维度
print('输出模型维度-->', model.get_dimension())

# 6、模型评估
print('模型评估-->', model.test(conf.process_test_datapath_word))
