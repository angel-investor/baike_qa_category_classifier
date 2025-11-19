# 导包操作
import fasttext
import jieba
from config import Config


# 加载配置
conf = Config()

# 加载模型
model = fasttext.load_model(conf.ft_model_save_path + "/model_word_2_auto_20251117.bin")


# 将模型封装成函数
def predict_func(data_dict):
    # 获取句子
    text = data_dict['text']
    # 进行分词
    text = " ".join(jieba.lcut(text))
    # 进行预测代码的编写
    re = model.predict(text)
    print('re-->', re)
    # 将预测结果添加到data_dict中，pred_class
    # data_dict['pred_class'] = re[0][0][9:]
    data_dict['pred_class'] = re[0][0].replace('__label__', '')
    return data_dict


if __name__ == '__main__':
    data = {"text": "电脑自动关机是怎么回事不一会儿就会自动关机 不一会儿就会自动关机"}
    result = predict_func(data)
    print('result-->', result)
