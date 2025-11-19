# 导入工具包
import fasttext
from config import Config
import datetime
import os

# 获取时间
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")
print("current_time-->", current_time)


# 加载配置
conf = Config()

# 1、模型训练
model = fasttext.train_supervised(input=conf.process_train_datapath_char,
                                  dim=100,  # 词向量的维度  默认100
                                  minn=1,  # 子词最小长度  默认0
                                  maxn=4,  # 子词最大长度  默认0
)
# 模型训练后信息:词向量信息,标签信息,词频信息和预测能力,子词信息等
print('词向量信息-->', model.get_word_vector("硬"))
print('标签信息-->', model.labels)
print('标签信息长度-->', len(model.labels))
# </s>:句子结束标记（sentence end）所以它的词频是样本数
print('(词, 词频)-->', model.get_words(include_freq=True))
print('(词, 词频)-->', list(zip(*model.get_words(include_freq=True))))
print('(词, 词频)表长度-->', len(list(zip(*model.get_words(include_freq=True)))))
print('-----------------------------------------------------')

# 2、模型保存
# model_path = config.ft_model_save_path + "/model_char_1_default.bin"
model_path = os.path.join(conf.ft_model_save_path, "model_char_1_default1.bin")
model.save_model(model_path)

# 3、模型预测
print('模型预测结果-->', model.predict("嘴 群 里 有 许 多 硬 结 是 啥   嘴 群 里 有 许 多 硬 结 是 啥"))

# 4、模型词表的查看
print('模型词表的查看-->', model.words[:10])

# 5、模型子词查看,注意:上述训练未开启子词
# < 和 > :子词模式中用于包裹词的边界符号
print('模型子词查看-->', model.get_subwords("硬结"))
# 输出模型维度
print('输出模型维度-->', model.get_dimension())

# 6、模型评估
# 结果(10000, 0.8634, 0.8634)
# 说明:(样本数,精确率,召回率)
print('模型评估-->', model.test(conf.process_test_datapath_char))
