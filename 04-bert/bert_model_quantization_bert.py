from a2_bert_classifer_model import BertClassifier
from config import Config
import torch
from a1_dataloader_utils import build_dataloader
from a3_train import model2dev

# 拓展: 打印当前支持的量化引擎，以了解环境中可用的量化计算后端
# print('engines-->', torch.backends.quantized.supported_engines)  # ['none', 'onednn', 'x86', 'fbgemm']
# 拓展: 设置上述支持量化引擎 ,注意:不是必须的,有默认操作,此处就是显式写出,让大家了解
# torch.backends.quantized.engine = 'fbgemm'

# 加载配置
conf = Config()

conf.device = "cpu"

# 准备数据
train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
# 准备模型
model = BertClassifier()
# 加载模型参数   因为当前用的动态量化,所以设备选择cpu
model.load_state_dict(torch.load(conf.model_save_path, map_location='cpu'))
# 模型评估模式
model.eval()
# 打印量化前的模型
# print("量化前模型-->",model)

# 量化前模型验证
report, f1score, accuracy, precision, recall = model2dev(model, dev_dataloader, conf.device)
print('report-->', report)
print("量化前f1score-->", f1score)

# 模型动态量化，指定线性层,同时指定类型qint8(默认)
# model：要量化的原始模型
# qconfig_spec：指定模型中哪些部分需要量化
# dtype：量化后的数据类型，默认就是qint8，
quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
print("量化后模型-->",quantized_model)

# 模型推理并评估
report2, f1score2, accuracy2, precision2, recall2 = model2dev(quantized_model, dev_dataloader, conf.device)
print('report2-->', report2)
print("量化后f1score-->", f1score2)

# 模型保存
torch.save(quantized_model.state_dict(), conf.bert_model_quantization_model_path)
print(f"模型已经保存，保存地址为：{conf.bert_model_quantization_model_path}")
