# 导入工具包
import fasttext
from config import Config
import datetime

conf = Config()

# 获取时间
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")

# 自动调参学习
# 1、模型训练
model = fasttext.train_supervised(input=conf.process_train_datapath_word,
                                  autotuneValidationFile=conf.process_dev_datapath_word,
                                  autotuneDuration=600,
                                  thread=8,
                                  seed=42,
                                  verbose=3
)
# 2、模型保存
# 保存路径
model_path = conf.ft_model_save_path + f"/model_word_2_auto_{current_time}.bin"
model.save_model(model_path)

# 3、模型预测
print('模型预测-->', model.predict("电脑 自动关机 是 怎么回事 不一会儿 就 会 自动关机   不一会儿 就 会 自动关机"))

# 4、模型词表的查看
print('词表-->', model.words[:10])

# 5、模型子词查看
print('子词-->', model.get_subwords("电脑"))
# 打印维度
print('维度-->', model.get_dimension())

# 6、模型评估
print(model.test(conf.process_test_datapath_word))
